def main():
	total_val = 0
	coke_price = 50
	
	while 0 < coke_price <= 50:
		coin = int(input("ENTER COINS: "))
		acc_rej = machine_check(coin)
		coke_price = cost(coin,total_val,coke_price)	
	else:
		if  total_val == coke_price or total_val > coke_price:
			print(f"CHANGE OWED: {total_val - coke_price}")
		
		
def cost(coin, total_val, coke_price):
	if machine_check(coin) is True:
		total_val += coin
		coke_price -= coin
		if coke_price > 0:
			print(f"AMOUNT DUE: {coke_price}")
		return coke_price
	elif coke_price >= 0:
		return coke_price

def machine_check(coin):
	if coin == 10 or coin == 25 or coin == 5:
		return True
	else:
		return False

main()

