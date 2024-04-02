def main():
	fruit = fruit_item()
	calories = cal(fruit)
	if calories == None:
		return
	else:
		print(f"CALORIES: {calories}")	

def fruit_item():
	fruit = input("ITEM: ").lower()
	return fruit

def cal(fruit):
	nutrition_facts = {
	"apple": 130,
	"avocado": 50,
	"banana": 110, 
	"cantaloupe": 50 ,
	"grapefruit" : 60,
	"grapes" : 90,
	"honeydew melon" : 50,
	"kiwifruit": 15,
	"lemon": 15,
	"lime": 20,
	"nectarine": 60,
	"orange": 80,
	"peach": 60,
	"pear": 100,
	"pineapple": 50,
	"strawberries": 50,
	"sweet cherries": 100,
	"tangerine": 50,
	"watermelon": 80,
	"plums": 70
}	
	calories = nutrition_facts.get(fruit)
	return calories

main()

