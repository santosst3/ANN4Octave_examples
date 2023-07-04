%%%%%%% MLP - ARQUIVO BASE %%%%%%%

%% Carregar arquivo de treinamento
clear all
close all
clc

cd MLP_examples

nentradas = 3; % Exemplo de aproximação
nintermed = 10;
nsaidas = 1;
dadostreinamento = csvread('treinamento_aproxima.csv');
dadosteste = csvread('teste_aproxima.csv');

##nentradas = 4; % Exemplo de classificação
##nintermed = 10;
##nsaidas = 3;
##dadostreinamento = csvread('treinamento_classifica.csv');
##dadosteste = csvread('teste_classifica.csv');

cd ..

xtreinamento = dadostreinamento(:,2:nentradas+1)'; % Nova sintaxe: func_forwardPMC já inclui bias -1
dtreinamento = dadostreinamento(:,nentradas+2:nentradas+nsaidas+1)';
xteste = dadosteste(:,2:nentradas+1)';
dteste = dadosteste(:,nentradas+2:nentradas+nsaidas+1)';

[maxinput,posmax] = max(xtreinamento');
[mininput,posmin] = min(xtreinamento');

% TO DO: Normalize inputs and outputs

for a = 1 : 1

  printf ("\n\n\nCASO %d\n\n",a);
  tic;

  %% Iniciar matrizes de pesos sinápticos e taxa de aprendizagem
  eta = 0.1; % Taxa de aprendizagem
  epsilon = 1e-6; % Tolerância
  alfa = 0.8; % Fator de momentum
  beta_log = 1; % Inclinação da f. logística
  rede_pmc = func_createANN('MLP',... % Tipo de rede utilizada
  [nentradas nintermed nsaidas],... % No de entradas/neurônios em cada camada
  {'logistic';'logistic'},... % Função de ativação em cada camada neural
  eta,epsilon,beta_log,alfa,...
  maxinput,mininput); % Máximos e mínimos dos dados de treinamento

  %% Treinamento PMC
  [rede_pmc,erro,epoca] = func_trainANN(rede_pmc,xtreinamento,dtreinamento);
  
  toc

  epoca
  printf ("\n\n\nMSE: %f\n\n",erro(end));
  figure
  plot(1:epoca,erro)
  set(gca,'fontsize',16,'FontName','LM Roman 12')
  xlabel('Número de épocas','fontsize',16,'Interpreter','LaTeX')
  ylabel('EQM','fontsize',16,'Interpreter','LaTeX')

  %% Fase de teste
  % Classificação de padrões
##  [yteste,yteste_pp,taxaacerto] = func_patternclass(rede_pmc,xteste,dteste);
##  matfinal.y{a} = yteste;
##  matfinal.acerto{a} = taxaacerto;
  
  % Aproximação de funções
  [yteste,erropercmed,varmed] = func_aproxfunc(rede_pmc,xteste,dteste);
  matfinal.y{a} = yteste;
  matfinal.parametros{a} = [erropercmed;varmed];

endfor
