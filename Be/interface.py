import json
import tkinter
from tkinter import *
from treino import extrair
from keras.models import load_model

model = load_model('model.h5')

condicao = json.loads(open('.\condicoes\intents.json').read())

base = Tk()
base.title("Be")
base.geometry("400x500") 
base.resizable(width=FALSE, height=FALSE)

def chatbot_response(msg):
    ints = extrair.class_prediction(msg, model)
    res = extrair.get_response(ints, condicao)
    return res

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        Chat.config(state=NORMAL)
        Chat.insert(END, f"Você: {msg}\n\n")
        Chat.config(foreground="#000000", font=("Arial", 12))

        response = chatbot_response(msg)
        Chat.insert(END, f"Be: {response}\n\n")

        Chat.config(state=DISABLED)
        Chat.yview(END)

Chat = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
Chat.config(state=DISABLED)

scrollbar = Scrollbar(base, command=Chat.yview)
Chat['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana", 10, 'bold'), text="Enviar", width="12", height=2, bd=0, bg="#666", activebackground="#333", fg='#ffffff', command=send)

EntryBox = Text(base, bd=0, bg="white", width="29", height="2", font="Arial")

scrollbar.place(x=376, y=6, height=386)
Chat.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=50, width=260)
SendButton.place(x=6, y=401, height=50)


base.mainloop()