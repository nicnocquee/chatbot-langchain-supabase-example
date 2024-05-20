import { promises as fs } from "fs";
import { Document } from "langchain/document";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { MarkdownTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import path from "node:path";
import dotenv from "dotenv";

dotenv.config({ path: path.join(__dirname, ".env.local") });

export const main = async () => {
  const documents = [];

  const splitter = new MarkdownTextSplitter({
    chunkSize: 100,
    chunkOverlap: 0,
  });

  const files = await fs.readdir(path.join(__dirname, "data"));
  // get json from directory
  for (const theFilePath of files) {
    if (!theFilePath.endsWith(".json") && !theFilePath.endsWith(".md")) {
      console.log(
        `Skipping ${theFilePath} because it's not a JSON or Markdown file`,
      );
      continue;
    }

    if (theFilePath.endsWith(".md")) {
      const filePath = path.join(__dirname, "data", theFilePath);
      const text = await fs.readFile(filePath, "utf8");
      const splitDocuments = await splitter.createDocuments([text]);
      documents.push(
        ...splitDocuments.map((doc) => ({
          ...doc,
          metadata: {
            ...doc.metadata,
            type: "troubleshooting",
          },
        })),
      );
    } else {
      const filePath = path.join(__dirname, "data", theFilePath);
      const json = await fs.readFile(filePath, "utf8");
      try {
        const data = JSON.parse(json);

        if (!Array.isArray(data)) {
          console.log(
            `Skipping ${theFilePath} because the content is not an array`,
          );
          continue;
        }

        for (const item of data) {
          const { question, answer, name, details } = item;
          if (question && answer) {
            documents.push(
              new Document({
                pageContent: `${question}\n${answer}`,
                metadata: {
                  filename: theFilePath,
                  type: "troubleshooting",
                },
              }),
            );
          } else if (name && details) {
            const {
              price,
              product_url,
              validity,
              image_url,
              type,
              category,
              group,
            } = item;

            documents.push(
              new Document({
                pageContent: [
                  `Product: ${name}`,
                  `Details: ${details}`,
                  price && price.length > 0 ? `Price: ${price}` : null,
                  category && category.length > 0
                    ? `Category: ${category}`
                    : null,
                  group && group.length > 0 ? `Group: ${group}` : null,
                  validity && validity.length > 0
                    ? `Validity: ${validity}`
                    : null,
                  image_url && image_url.length > 0
                    ? `Image URL: ${image_url}`
                    : null,
                  product_url && product_url.length > 0
                    ? `Product URL: ${product_url}`
                    : null,
                ]
                  .filter(Boolean)
                  .join("\n"),
                metadata: {
                  filename: theFilePath,
                  product: item,
                  topic: type,
                  price:
                    parseInt(
                      price
                        ?.replace("Rp", "")
                        .replace(/\"/g, "")
                        .replace(/\./g, ""),
                    ) || 0,
                  type: "product,",
                },
              }),
            );
          } else {
            console.log(
              `Skipping ${theFilePath} because it doesn't have a question and answer`,
            );
            break;
          }
        }
      } catch (error) {
        console.log(`Skipping ${theFilePath} because it's not valid JSON`);
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
