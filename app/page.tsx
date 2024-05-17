import { ChatWindow } from "@/components/chat-window";

export default function AgentsPage() {
  const InfoCard = (
    <div className="p-4 md:p-8 rounded bg-[#25252d] w-full max-h-[85%] overflow-hidden">
      <h1 className="text-3xl md:text-4xl mb-4">Smartfren Customer Care</h1>
      <p>Ask me anything about your Smartfren account!</p>
    </div>
  );
  return (
    <ChatWindow
      endpoint="api/chat/retrieval"
      emptyStateComponent={InfoCard}
      showIngestForm={false}
      placeholder={"What can I help you with?"}
      emoji="🤖"
      titleText="Smartfren Customer Care"
    ></ChatWindow>
  );
}
