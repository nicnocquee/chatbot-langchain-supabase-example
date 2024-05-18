import { promises as fs } from "fs";
import { Document } from "langchain/document";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { MarkdownTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import path from "node:path";
import dotenv from "dotenv";

dotenv.config({ path: path.join(__dirname, ".env.local") });

const knowledgeDocuments = [
  { path: "addons.json", type: "product", topic: "addons" },
  {
    path: "aktifkan-paket-internet.md",
    type: "troubleshooting",
    topic: "activating-internet-package",
  },
  { path: "broadbands.json", type: "product", topic: "broadbands" },
  { path: "bundles.json", type: "product", topic: "bundles" },
  {
    path: "ganti-nomor.json",
    type: "troubleshooting",
    topic: "changing-number",
  },
  {
    path: "internet-lemot.md",
    type: "troubleshooting",
    topic: "slow-internet",
  },
  { path: "low-signal.md", type: "troubleshooting", topic: "low-signal" },
  { path: "minta-pulsa.json", type: "product", topic: "minta-pulsa" },
  { path: "packages.json", type: "product", topic: "packages" },
  {
    path: "paket-kuota-100GB.json",
    type: "product",
    topic: "paket-kuota-100GB",
  },
  {
    path: "paket-kuota-200GB.json",
    type: "product",
    topic: "paket-kuota-200GB",
  },
  { path: "paket.json", type: "product", topic: "paket" },
  { path: "power-up.json", type: "product", topic: "power-up" },
  { path: "pulsa-darurat.json", type: "product", topic: "pulsa-darurat" },
  { path: "reaktivasi.json", type: "troubleshooting", topic: "reaktivasi" },
  { path: "registrasi.json", type: "troubleshooting", topic: "registration" },
  { path: "sptourist.json", type: "troubleshooting", topic: "tourists" },
  {
    path: "touriststaterpack.json",
    type: "troubleshooting",
    topic: "tourists",
  },
  { path: "unlimited.json", type: "troubleshooting", topic: "unlimited" },
  {
    path: "voucher-data-super-kuota.json",
    type: "troubleshooting",
    topic: "voucher-data-super-kuota",
  },
  {
    path: "voucher-data-unlimited.json",
    type: "troubleshooting",
    topic: "voucher-data-unlimited",
  },
];

export const main = async () => {
  const documents = [];

  const splitter = new MarkdownTextSplitter({
    chunkSize: 100,
    chunkOverlap: 0,
  });

  // get json from directory
  for (const knowledge of knowledgeDocuments) {
    const { path: theFilePath, type, topic } = knowledge;
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
            topic,
            type,
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
                pageContent: `About ${theFilePath.replace(
                  ".json",
                  "",
                )}: ${question}\n${answer}`,
                metadata: {
                  filename: theFilePath,
                  topic,
                  type,
                },
              }),
            );
          } else if (name && details) {
            const { price, product_url, validity, image_url } = item;
            documents.push(
              new Document({
                pageContent: [
                  `Product: ${name}`,
                  price && price.length > 0 ? `Price: ${price}` : null,
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
                  topic,
                  type,
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
