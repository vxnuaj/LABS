def main():
        ui = user_input()
        strip_ui = vowel_strip(ui)
        print(strip_ui)
        return


def user_input():
        ui = input("ENTER YOUR TWEET: ")
        return ui

def vowel_strip(ui):
        strip_ui = ui.replace("a", "")
        strip_ui = strip_ui.replace("e", "")
        strip_ui = strip_ui.replace("i", "")
        strip_ui = strip_ui.replace("o", "")
        strip_ui = strip_ui.replace("u", "")
        return strip_ui

main()
-- INSERT --
