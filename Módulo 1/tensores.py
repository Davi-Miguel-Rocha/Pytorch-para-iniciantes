

import torch

# Criando dois tensores
a = torch.tensor([2,3,4])

b = torch.tensor([3,1,2])

# Operações básicas
soma = a + b

mult = a * b

sub = a - b

div = a / b

# Multiplicação por escalar (por um número)
escalar_A = a * 3

escalar_B = b * 3


# Imprimindo os resultados
print(f'A soma dos tensores resulta em: {soma}\n')

print(f'A multiplicação dos tensores resulta em: {mult}\n')

print(f'A subtração resulta em: {sub}\n')

print(f'A divisão dos tensores resulta em: {div}\n')

print(f'Multiplicando A pelo escalar 3 resultará em: {escalar_A}\n')

print(f'Multiplicando B pelo escalar 3 resultará em: {escalar_B}\n')