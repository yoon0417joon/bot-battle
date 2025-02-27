
# bot battle
**by - I don't care**

this is a program inspired by **Robert Axelrod**'s experiment.


# Authors

- [danmu1ji](https://github.com/yoon0417joon)
- [chatgpt](https://chat.openai.com)
- [grok](https://grok.com/)
- [stackoverflow](https://stackoverflow.com/) (of course)


# Demo
![alt text](https://i.ibb.co/Txwrxtxr/2025-02-27-143533.png)
# Feedback

SHUT UP for a moment. Gonna make a page of it with **GTA 7 official release. **~~after I die?~~


# Installation

I don't fucking know how to make it bruh

```textbox
    tag --> (version)
    download sourse_code.zip
    unzip it
    run botbattle.py
```
    
# Features

- dark mode (no nerds use light mode)
- addon



# How to make an addon?

bot's name and description is made by bot_decorator.

```text
def bot_decorator(name, description):
def decorator(func):
    def wrapper(input_value, *args, **kwargs):
        if isinstance(input_value, str):
            if input_value.lower() == "name":
                return name
            elif input_value.lower() == "description":
                return description
        return func(input_value, *args, **kwargs)
    wrapper.bot_name = name
    wrapper.bot_description = description
    return wrapper
return decorator
```
using this, you can make bots name and description.(optional)

bot will get input 1 or 2 depening on its side.

given datas are:
- b1log (list) 1st bot's log
- b2log (list) 2nd bot's log
- b1point (int) 1st bot's point
- b2point (int) 2nd bot's point
- round (int) rounds(starting at 0)

and, the bot itself **has to return "trust" or "betray"**

**example**

    @bot_decorator("tit-for-tat Bot", "a simple bot that trusts on the first move and 
    then mimics the opponent.")
    def t4t(input):
        if round == 0:
            return "trust"
        if input == 1:
            return b2log[-1]
        else:
            return b1log[-1]

after making every bots you want, you have to put their names in specific list depending on its usage.

for bot's functions, it goes to

    bots = []

and for functions that isn't a bot, it goes to

    exceptions = []

**plus**, the libraries used should go to

    required_libraries = []

## packaging

- addon's name will be the folder's name. (yes it uses folder for addons)
- python file's name should be "addon.py".
- bot images should be located in "(addon_name)\assets" with naming "(bot_name).png"
# Roadmap

- fullscreen (v0.2)
- browser support (v0.8)
- android support (v1.0 release)
- ~~ios support~~ (I don't have an iphone)



# Optimizations

No. This is SHIT.


# Run Locally

**required libraries are :**
- sys
- os
- math
- random
- itertools
- inspect
- statistics
- textwrap
- ast
- subprocess
- importlib
- mplcursors
- matplotlib
- PyQt6
- qt-material
