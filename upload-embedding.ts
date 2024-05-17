import { promises as fs } from "fs";
import { Document } from "langchain/document";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import path from "node:path";
import dotenv from "dotenv";

dotenv.config({ path: path.join(__dirname, ".env.local") });

export const main = async () => {
  const documents = [];

  const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
    chunkSize: 256,
    chunkOverlap: 20,
  });

  // get json from directory
  const files = await fs.readdir(path.join(__dirname, "data"));
  for (const file of files) {
    if (!file.endsWith(".json") && !file.endsWith(".md")) {
      console.log(`Skipping ${file} because it's not a JSON or Markdown file`);
      continue;
    }

    if (file.endsWith(".md")) {
      const filePath = path.join(__dirname, "data", file);
      const text = await fs.readFile(filePath, "utf8");
      const splitDocuments = await splitter.createDocuments([text]);
      documents.push(...splitDocuments);
    } else {
      const filePath = path.join(__dirname, "data", file);
      const json = await fs.readFile(filePath, "utf8");
      try {
        const data = JSON.parse(json);

        if (!Array.isArray(data)) {
          console.log(`Skipping ${file} because the content is not an array`);
          continue;
        }

        for (const item of data) {
          const { question, answer } = item;
          if (question && answer) {
            documents.push(
              new Document({
                pageContent: `About ${file.replace(
                  ".json",
                  "",
                )}: ${question}\n${answer}`,
                metadata: { filename: file, question },
              }),
            );
          } else {
            console.log(
              `Skipping ${file} because it doesn't have a question and answer`,
            );
            break;
          }
        }
      } catch (error) {
        console.log(`Skipping ${file} because it's not valid JSON`);
      }
    }
  }

  if (documents.length === 0) {
    console.log("No documents found");
    return;
  }

  const client = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_PRIVATE_KEY!,
  );

  console.log("Deleting existing embeddings");
  const result = await client.from("documents").delete().neq("id", -1);

  console.log("Creating and uploading embeddings");
  await SupabaseVectorStore.fromDocuments(
    documents,
    new OpenAIEmbeddings({
      model: process.env.EMBEDDING_MODEL_NAME!,
    }),
    {
      client,
      tableName: "documents",
      queryName: "match_documents",
    },
  );

  console.log("Uploaded embeddings");
};

main();
