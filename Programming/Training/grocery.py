def main():
    groceries = {} #Initializing a dictionary to hold our groceries
    while True: #Initializing infinite loop until CTRL + D
        try: #While the infinite loop is running, try this: get input from user (so continuously ask for input until CTRL + D)
            item = input() #get user input
            groceries = add_item(item,groceries) #runs the add_items function, which we can add an item to the list or increment to a previous one
        except EOFError: #When EOFError of CTRL+D
            sorted_groceries = sorted(groceries.items()) #Sort the items in the groceries dictionary alphabetically.
            for item, count in sorted_groceries: # for each item (key) and count (value), in the sorted_groceries dictionary (sorted alphabetically)
                 print(count, item.upper()) #print the count (value) of the dictionary and the item (key).
            break #end the inf loop.

def add_item(item, groceries): #takes in specific item and the groceries dict to append an input item into the groceries
     if item in groceries: #if the item is already existing within the groceries dict,
          groceries[item] += 1 #then increment 1 to the value of the item, indicating that we're purchasing mroe than 1 of the item.
          return groceries #return the updated dict.
     else: #otherwise
          groceries[item] = 1 # add the item to the dict and assign it a value of 1, to indicate that we only have 1 instance of the item in our dict as of yet.
          return groceries #return the updated dict.

main()
