import { NextRequest, NextResponse } from "next/server";
import { Message as VercelChatMessage, StreamingTextResponse } from "ai";

import { createClient } from "@supabase/supabase-js";
import { AttributeInfo } from "langchain/schema/query_constructor";
import { ChatOpenAI, OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { Document } from "@langchain/core/documents";
import { RunnableSequence, RunnableBranch } from "@langchain/core/runnables";
import {
  BytesOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";
import { SupabaseTranslator } from "langchain/retrievers/self_query/supabase";

export const runtime = "edge";

const combineDocumentsFn = (docs: Document[]) => {
  const serializedDocs = docs.map((doc) => doc.pageContent);
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

const classificationPrompt =
  PromptTemplate.fromTemplate(`Given the user question and chat history below, classify it as either being about \`troubleshooting\`, \`product\`, or  \`product_search_by_price\`.

Respond \`troubleshooting\` when the user is asking for help with a problem or how to do something.

Respond \`product\` when the user is asking for information about a product or service, or a list of products.

Respond \`product_search_by_price\` when the user is asking for information about a product or service and is looking for a specific price.

Do not respond with more than one word.

<question>
{question}
</question>

<chat_history>
  {chat_history}
</chat_history>

Make sure you consider the chat history when classifying the question.

Classification:`);

const CONDENSE_QUESTION_TEMPLATE = `Given the chat history and a follow up question, rephrase the follow up question to be a standalone question that includes clear and informative subject, object, and other information, in its original language.

<chat_history>
  {chat_history}
</chat_history>

Follow Up Input: {question}
Standalone question:`;
const condenseQuestionPrompt = PromptTemplate.fromTemplate(
  CONDENSE_QUESTION_TEMPLATE,
);

const ANSWER_TEMPLATE = `
You are a smartfren customer care agent. Answer the question in the same language as the question based only on the following context and chat history:

<context>
  {context}
</context>

<chat_history>
  {chat_history}
</chat_history>


If the answer is not contained in the context, apologize that you cannot help with the inquiry and ask if they have any questions about Smartfren.

If the user complains or asks for help, show some sympathy and apologize for the inconvenience first then answer the question. Otherwise, just answer the question.

If the user asks for a product or a list of products, and you can find it in the context, then answer the question. But if the products is not in the context, recommend a product that you know is available based on the context.

Be casual. Don't sound too formal.

Question: {question}
`;
const answerPrompt = PromptTemplate.fromTemplate(ANSWER_TEMPLATE);

const ANSWER_PRODUCT_SEARCH_BY_PRICE_TEMPLATE = `
You are a smartfren customer care agent. User asked the following question:

<question>
{question}
</question>

The possible products that the user is looking for is:

<product>
{context}
</product>

Give an answer in the same language as the question to the user based on the products information. 
Be casual. Don't sound too formal.

`;
const answerProductSearchByPricePrompt = PromptTemplate.fromTemplate(
  ANSWER_PRODUCT_SEARCH_BY_PRICE_TEMPLATE,
);

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

    // Classification chain
    const classificationChain = RunnableSequence.from([
      classificationPrompt,
      model,
      new StringOutputParser(),
    ]);

    const answerChain = RunnableSequence.from([
      {
        context: (input) => {
          const { question, topic } = input;
          const retriever = vectorstore.asRetriever({
            verbose: true,
            filter: { type: topic },
            callbacks: [
              {
                handleRetrieverEnd(documents) {
                  resolveWithDocuments(documents);
                },
              },
            ],
          });

          return retriever.pipe(combineDocumentsFn).invoke(question);
        },
        chat_history: (input) => input.chat_history,
        question: (input) => input.question,
      },
      answerPrompt,
      model,
    ]);

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
      callbacks: [
        {
          handleRetrieverEnd(documents) {
            resolveWithDocuments(documents);
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

    const answerProductSearchByPriceChain = RunnableSequence.from([
      {
        context: (input) =>
          selfQueryRetriever.pipe(combineDocumentsFn).invoke(input.question),
        chat_history: (input) => input.chat_history,
        question: (input) => input.question,
      },
      answerProductSearchByPricePrompt,
      model,
    ]);

    const branch = RunnableBranch.from([
      [
        (x: { topic: string; question: string; chat_story: string }) =>
          x.topic.toLowerCase().includes("product_search_by_price"),
        (input) => answerProductSearchByPriceChain.invoke(input),
      ],
      (x: { topic: string; question: string; chat_story: string }) =>
        answerChain.invoke(x),
    ]);

    const conversationalRetrievalQAChain = RunnableSequence.from([
      {
        question: standaloneQuestionChain,
        chat_history: (input) => input.chat_history,
      },
      {
        question: (input) => input.question,
        topic: classificationChain,
        chat_history: (input) => input.chat_history,
      },
      branch,
      new BytesOutputParser({ verbose: true }),
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

    return new StreamingTextResponse(stream, {
      headers: {
        "x-message-index": (previousMessages.length + 1).toString(),
        "x-sources": serializedSources,
      },
    });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}
