import type { Message } from "ai/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function ChatMessageBubble(props: {
  message: Message;
  aiEmoji?: string;
  sources: any[];
}) {
  const colorClassName =
    props.message.role === "user" ? "bg-sky-600" : "bg-slate-50 text-black";
  const alignmentClassName =
    props.message.role === "user" ? "ml-auto" : "mr-auto";
  const prefix = props.message.role === "user" ? "🧑" : props.aiEmoji;
  return (
    <div
      className={`${alignmentClassName} ${colorClassName} rounded px-4 py-2 max-w-[80%] mb-8 flex flex-row space-x-2`}
    >
      <div>{prefix}</div>
      <div className="w-full flex flex-col flex-1">
        <ReactMarkdown remarkPlugins={[remarkGfm]} className="prose">
          {props.message.content}
        </ReactMarkdown>
      </div>
    </div>
  );
}
