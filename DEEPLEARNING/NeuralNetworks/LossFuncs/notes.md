### General

- Always thoroughly go through your math and derive the equations for yourself when you notice something looks off. YUou shoudl be able to do it on your own. You can't understand what you can't break on your own. You have to know the first principles

### Implememting MAE and MSE

- The derivative of MAE is -np.sign(y - pred) with an negative in front bc of the Chain Rule. Important! Make sure to thoroughly go through your math before making assumptions that something else is wrong.
- Standardizing data is nice! `from sklearn.preprocessing import StandardScaler`

### MAE vs MSE
I think it's clear that MSE punishes more heavily than MAE. The loss here only goes to about .000X while in the MSE implementation, the loss goes to 0.0000000000000000000000000000086602. 

This is very much due to the squared nature of the MSE.
