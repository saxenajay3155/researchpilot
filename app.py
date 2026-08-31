#===============================================
#Gradio UI
import gradio as gr

from retrieval import question_answer

def evidence_format(list_of_contexts, list_of_metadatas):
    evidence = ""
    for i in range(len(list_of_contexts)):
        evidence += f"""## Evidence {i+1}
        
**Source:** {list_of_metadatas[i]["source"]}

{list_of_contexts[i]}


---

"""
    return evidence

def respond(question, history):
    history = history or []
    answer, contexts, metadatas = question_answer(question)
    history.append({"role": "user","content": question})
    history.append({"role": "assistant","content": answer})
    evidence = evidence_format(contexts, metadatas)
    return history, evidence, ""


with  gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# Research Pilot \n Ask questions about the annual reports of Infosys, ITC and TCS. Answers are generated from retrieved report evidence.")
        # gr.Markdown("Ask questions about the annual reports of Infosys, ITC and TCS. Answers are generated from retrieved report evidence.")

    with gr.Row():
        
        with gr.Column(scale=1.25):
            gr.Markdown("## Research Assistant")
            chatbot = gr.Chatbot()
            question_box = gr.Textbox(placeholder="Ask question here")
            submit_button = gr.Button("Send")
            
        with gr.Column(scale=1):
            gr.Markdown("## Supporting Evidence")
            evidence_box = gr.Markdown("Evidence goes here.", height=550)

        submit_button.click(fn=respond, inputs=[question_box, chatbot], outputs=[chatbot, evidence_box ,question_box])
        question_box.submit(fn=respond, inputs=[question_box, chatbot], outputs=[chatbot, evidence_box ,question_box])
if __name__ == "__main__":
    demo.launch()