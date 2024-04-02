names = []

for i in range(3):
    names.append(input("What's your name? "))

for name in sorted(names):
    print(f"Hello {name}!")

with open("names.txt", "w") as f:
    for name in sorted(names):
        f.write(f"{name}\n")
