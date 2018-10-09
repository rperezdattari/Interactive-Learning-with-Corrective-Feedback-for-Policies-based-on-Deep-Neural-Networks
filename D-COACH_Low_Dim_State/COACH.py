from linear_RBFs import linear_RBFs
from NN import NN
from Teacher import NN as teacher


#Function in charge of selecting the approximator
def COACH(agent, buffer=True):
    if agent == 'NN':
        return NN(buffer)
    elif agent == 'linear_RBFs':
        return linear_RBFs()
    elif agent == 'Teacher':
        return teacher()
    else:
        raise NameError('The selected agent is not valid. Try using: NN or linear_RBFs')
