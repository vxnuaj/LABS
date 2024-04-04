def main():
    names = []
    while True:
        try:
            name = input("Name: ")
            add_name(names, name)
        except EOFError:
            if len(names) == 1:
                print(f"Adieu, adieu, to {names[0]}")
            elif len(names) == 2:
                print(f"Adieu, adieu, to {', '.join(names[:-1])} and {names[-1]}")
            elif len(names) > 2:
                print(f"Adieu, adieu, to {', '.join(names[:-1])}, and {names[-1]}")
            break
    
def add_name(names, name):
    names.append(name)
    return names

main()