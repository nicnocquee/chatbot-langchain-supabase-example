# Chatbot Langchain Nextjs Example

Example chatbot that will only answer questions based on the data in the `data` folder. The data can be in JSON or Markdown format. When the file is in JSON format, the content should be an array of objects with a `question` and `answer` property.

# Getting Started

Requirements:

- create a Supabase account
- create a Supabase project
- Run the following query in Supabase SQL Editor:

```sql
-- Enable the pgvector extension to work with embedding vectors
create extension vector;

-- Create a table to store your documents
create table documents (
  id bigserial primary key,
  content text, -- corresponds to Document.pageContent
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(1536) -- 1536 works for OpenAI embeddings, change if needed
);

-- Create a function to search for documents
create function match_documents (
  query_embedding vector(1536),
  match_count int DEFAULT null,
  filter jsonb DEFAULT '{}'
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  embedding jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    (embedding::text)::jsonb as embedding,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

- clone this repo
- set up environment variables in your repo's `.env.local` file. Copy the `.env.example` file to `.env.local`.
- add your OpenAI API key in the `.env.local` file.
- add your Supabase private service key in the `.env.local` file.
- add your Supabase URL in the `.env.local` file.
- Run `npx tsx upload-embedding.ts` to delete existing embeddings from Supabase, create embeddings, and upload the data to Supabase.
- `bun install`
- `bun dev`
- Open [http://localhost:3000](http://localhost:3000) with your browser to see the result! Ask the bot something and you'll see a streamed response:

Notes:

- The model to use for the chatbot can be set in the `MODEL_NAME` environment variable. Defaults to `gpt-3.5-turbo`.
- The model to use for the embeddings and retrieval can be set in the `EMBEDDING_MODEL_NAME` environment variable. Defaults to `text-embedding-3-small`.
