from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from pinecone import Pinecone
from groq import Groq
import os

app = Flask(__name__)

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index("keyword-intel")
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

def embed_query(text):
    result = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[text],
        parameters={"input_type": "query"}
    )
    return result[0]['values']

def search_pinecone(query, top_k=5):
    vector = embed_query(query)
    namespaces = [
        "b92b1c18-bacf-4e66-b68e-cd9a2773a43f",
        "negatives_1ab483a7-b766-483e-945f-98bf883cb889",
        "negatives_c0118566-ee85-468a-a6b0-009fd5f5728c",
        "anchors"
    ]
    all_matches = []
    for ns in namespaces:
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=ns
        )
        all_matches.extend(results['matches'])
    all_matches.sort(key=lambda x: x['score'], reverse=True)
    return all_matches[:top_k]

def build_context(matches):
    context_parts = []
    for i, match in enumerate(matches):
        meta = match.get('metadata', {})
        score = round(match.get('score', 0), 4)
        context_parts.append(f"{i+1}. {meta} (relevance: {score})")
    return "\n".join(context_parts)

def get_answer(question, context):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to a keyword database. "
                    "Answer the user's question using the retrieved records below. "
                    "Be concise and direct.\n\n"
                    f"Retrieved records:\n{context}"
                )
            },
            {"role": "user", "content": question}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').strip()
    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("Please send a question!")
        return str(resp)

    try:
        matches = search_pinecone(incoming_msg)
        if not matches:
            msg.body("I couldn't find anything relevant in the database.")
            return str(resp)

        context = build_context(matches)
        answer = get_answer(incoming_msg, context)
        msg.body(answer)
    except Exception as e:
        msg.body(f"Error: {str(e)}")

    return str(resp)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)