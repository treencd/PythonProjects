from guizero import App, Text, PushButton, Box, TextBox
import random

item_list = []
old_value = ""
global win_height
win_height = 250

def randomizer(result):
    food = random.choice(range(len(item_list)))
    choice = item_list[food]

    print(item_list[food])
    result.visible = True
    result.value = item_list[food]
    result.textcolor = "green"
    result.textsize = 20

def update(data):
    if data.key == " ":
        global win_height
        item_list.append(InputBox.value)
        # print(item_list)
        InputBox.clear()
        i = len(item_list)
    # else: print(data.key)

FoodApp = App(title="Randomizer", layout="auto", height=win_height, width=640, bg="#424242",visible=True)

Box(FoodApp,width=10,height=25)
Instruction = Text(FoodApp,"Type one of the options and press 'space'",size=20,color="white")
Box(FoodApp,width=10,height=25)
InputBox = TextBox(FoodApp)
InputBox.when_key_pressed = update
Box(FoodApp,width=100,height=5)
answer = TextBox(FoodApp)
answer.bg="#424242"
answer.textcolor="white"
answer.visible = False

done = PushButton(FoodApp, command=randomizer, args=[answer],text="Randomize")

FoodApp.display()

