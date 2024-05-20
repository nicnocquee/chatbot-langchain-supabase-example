import { NextRequest, NextResponse } from "next/server";
import {
  Message as VercelChatMessage,
  StreamingTextResponse,
  LangChainAdapter,
} from "ai";

import { createClient } from "@supabase/supabase-js";
import { AttributeInfo } from "langchain/schema/query_constructor";
import { ChatOpenAI, OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { Document } from "@langchain/core/documents";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";
import { SupabaseTranslator } from "langchain/retrievers/self_query/supabase";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";

export const runtime = "edge";

const combineDocumentsFn = (docs: Document[]) => {
  let serializedDocs = docs.map((doc) => doc.pageContent);
  return serializedDocs.join("\n\n");
};

const formatVercelMessages = (chatHistory: VercelChatMessage[]) => {
  const formattedDialogueTurns = chatHistory.map((message) => {
    if (message.role === "user") {
      return `Human: ${message.content}`;
    } else if (message.role === "assistant") {
      return `Assistant: ${message.content}`;
    } else {
      return `${message.role}: ${message.content}`;
    }
  });
  return formattedDialogueTurns.join("\n");
};

const CONDENSE_QUESTION_TEMPLATE = `Given the chat history and a follow up question, rephrase the follow up question to be a standalone question that includes clear and informative subject, object, and other information, in its original language. Remember to always rephrase in the same language as the question.

Note that the follow up input could be unrelated to the chat history, or it could be a question that is not related to the chat history. For example, user might ask something unrelated to previous questions. 

<chat_history>
  {chat_history}
</chat_history>

Follow Up Input: {question}

If the follow up input is not a question, or cannot be interpreted as a question, respond with "[NOT_QUESTION] {question}".

Standalone question:`;
const condenseQuestionPrompt = PromptTemplate.fromTemplate(
  CONDENSE_QUESTION_TEMPLATE,
);

const ANSWER_TEMPLATE = `
You are SFChat, a chatbot designed to assist users with various needs related to Smartfren. You are trained using an extensive database about Smartfren's <context> and capable of providing information and answering questions about Smartfren's products and services.

Smartfren is one of the fastest-growing telecommunications companies in Indonesia. Smartfren is a pioneer of 4G LTE internet services and VoLTE services.

<context>
{context}
</context>

<time>
{time}
</time>

<chat_history>
  {chat_history}
</chat_history>

Carefully review the provided context. If a user's question is unrelated to the given <context>, be firm in stating that you can only respond to questions related to Smartfren and provide a polite response. Sometimes the context and the question may not match. Always answer based on the context, and do not use your own knowledge.

When a user initiates a conversation, greet them warmly according to the provided <time>. For example, if <time> indicates morning, use an appropriate morning greeting. If it indicates afternoon or evening, use a greeting suitable for that time of day.

Make sure to provide answers that are relevant to the given <time>. If a user wants to purchase a product or service, ensure the information provided matches the relevant <time> period.

Recognize the language used by the user and respond in the same language.

Show empathy and maintain a friendly yet professional communication style. Avoid using complicated technical terms and lengthy explanations. If the user is experiencing an issue, show understanding and a commitment to help them as quickly as possible.

Proactively offer relevant recommendations, such as other similar packages or services. Always strive to present specific additional products or services that might interest the user, providing detailed information about these options without waiting for the user to ask.

When responding, highlight the benefits and unique features of recommended products or services, and create a sense of urgency by mentioning any ongoing promotions or limited-time offers. Tailor your recommendations to the user's needs and preferences to make the upsell more appealing.

If the question starts with [NOT_QUESTION], then reply appropriately.

Question: {question}
`;
const answerPrompt = PromptTemplate.fromTemplate(ANSWER_TEMPLATE);

/**
 * This handler initializes and calls a retrieval chain. It composes the chain using
 * LangChain Expression Language. See the docs for more information:
 *
 * https://js.langchain.com/docs/guides/expression_language/cookbook#conversational-retrieval-chain
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const previousMessages = messages.slice(0, -1);
    const currentMessageContent = messages[messages.length - 1].content;

    const model = new ChatOpenAI({
      modelName: process.env.MODEL_NAME,
      temperature: 0.2,
      verbose: true,
    });

    const client = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_PRIVATE_KEY!,
    );
    const vectorstore = new SupabaseVectorStore(
      new OpenAIEmbeddings({
        model: process.env.EMBEDDING_MODEL_NAME!,
      }),
      {
        client,
        tableName: "documents",
        queryName: "match_documents",
      },
    );

    /**
     * We use LangChain Expression Language to compose two chains.
     * To learn more, see the guide here:
     *
     * https://js.langchain.com/docs/guides/expression_language/cookbook
     *
     * You can also use the "createRetrievalChain" method with a
     * "historyAwareRetriever" to get something prebaked.
     */
    const standaloneQuestionChain = RunnableSequence.from([
      condenseQuestionPrompt,
      model,
      new StringOutputParser({ verbose: true }),
    ]);

    let resolveWithDocuments: (value: Document[]) => void;
    const documentPromise = new Promise<Document[]>((resolve) => {
      resolveWithDocuments = resolve;
    });
    let resolveWithProducts: (value: Document[]) => void;
    const productsPromise = new Promise<Document[]>((resolve) => {
      resolveWithProducts = resolve;
    });

    const attributeInfo: AttributeInfo[] = [
      {
        name: "price",
        description: "Price of the product",
        type: "number",
      },
    ];

    const selfQueryRetriever = SelfQueryRetriever.fromLLM({
      llm: new OpenAI(),
      vectorStore: vectorstore,
      documentContents: "Summary of the product",
      attributeInfo,
      searchParams: {
        filter: {
          type: "product",
        },
      },
      callbacks: [
        {
          handleRetrieverEnd(documents) {
            resolveWithProducts(documents);
          },
        },
      ],
      /**
       * We need to use a translator that translates the queries into a
       * filter format that the vector store can understand. LangChain provides one here.
       */
      structuredQueryTranslator: new SupabaseTranslator(),
      verbose: true,
    });

    const answerChain = RunnableSequence.from([
      {
        context: (input) => {
          const { question } = input;
          const retriever = MultiQueryRetriever.fromLLM({
            llm: model,
            retriever: vectorstore.asRetriever({
              verbose: true,
              k: 10,
              searchType: "similarity",
              callbacks: [
                {
                  handleRetrieverEnd(documents) {
                    resolveWithDocuments(documents);
                  },
                },
              ],
            }),
            verbose: true,
          });

          return retriever.pipe(combineDocumentsFn).invoke(question);
        },
        chat_history: (input) => input.chat_history,
        question: (input) => input.question,
        time: () => new Date().getTime(),
      },
      answerPrompt,
      model,
    ]);

    const conversationalRetrievalQAChain = RunnableSequence.from([
      {
        question: standaloneQuestionChain,
        chat_history: (input) => input.chat_history,
      },
      answerChain,
    ]);

    const stream = await conversationalRetrievalQAChain.stream({
      question: currentMessageContent,
      chat_history: formatVercelMessages(previousMessages),
    });

    const documents = await documentPromise;
    const serializedSources = Buffer.from(
      JSON.stringify(
        documents.map((doc) => {
          return {
            pageContent: doc.pageContent,
            metadata: doc.metadata,
          };
        }),
      ),
    ).toString("base64");

    const aiStream = LangChainAdapter.toAIStream(stream);

    return new StreamingTextResponse(aiStream);
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}
