import torch

import torch.nn as nn

import torch.optim as optim


# Criando a rede
class RedeSoma(nn.Module):

    def __init__(self, cam_entrada, cam_oculta, cam_saida):
        super(RedeSoma, self).__init__()

    
        self.fc1= nn.Linear(cam_entrada,cam_oculta)

        self.fc2= nn.Linear(cam_oculta,cam_saida)

        

    def forward(self, x):

        x = torch.relu(self.fc1(x))

        x = self.fc2(x)

        return x
    
entradas = torch.tensor([[2.0, 2.0],
                         [4.0, 4.0],
                         [8.0, 8.0],
                         [16.0, 16.0]])

saidas = torch.tensor([[4.0],
                       [8.0],
                       [16.0],
                       [32.0]])


input_camada=2

hidden_camada=10

saida_camada=1


Modelo=RedeSoma(input_camada,hidden_camada,saida_camada)

criterio=nn.MSELoss()

otimizador=optim.Adam(Modelo.parameters(), lr=0.1)


for epoch in range(561):

    saida_predita = Modelo(entradas)

    perda= criterio(saida_predita,saidas)

    otimizador.zero_grad()

    perda.backward()

    otimizador.step()

    if epoch % 100 == 0:
        print(f'Epoch:{epoch} | Perda: {perda.item():.4f}')



with torch.no_grad():

    novos_dados = torch.tensor([[2.0, 4.0],
                                [4.0, 2.0]])
    
    pred= Modelo(novos_dados)

    print('Previsões:')

    print(pred)
