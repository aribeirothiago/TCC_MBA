# Processamento de Linguagem Natural e Análise de Sentimentos em negociações na abertura da B3
## Thiago de Almeida Ribeiro e Adâmara Santos Gonçalves Felício
### Resumo
Quando ocorre uma diferença entre os preços de fechamento e abertura de uma ação, diz-se que há um “gap de abertura”. Esse fenômeno normalmente está ligado à diferença de informações disponíveis em cada um dos momentos, de forma que notícias podem ajudar a prever se tal “gap” será fechado ou não. Neste trabalho, foi construída uma ferramenta que indica, com base nos preços de fechamento e abertura de ações da B3, além de manchetes de notícias disponíveis, aquelas que devem ser compradas no momento de abertura do mercado e quando vendê-las, de forma a obter lucro. Para analisar as manchetes de forma automática, foi utilizado o Processamento de Linguagem Natural [PLN], ou, mais especificamente, a Análise de Sentimentos, que, por sua vez, pode fazer uso de diversas técnicas. Elas são divididas em abordagem não supervisionada, da qual faz parte o Léxico para Inferência Adaptada [LeIA], e abordagem supervisionada, na qual se encaixam as técnicas de Aprendizado de Máquina, como Random Forest e Naive Bayes. Utilizando o período de um ano e a porcentagem de acerto, ou precisão, como métrica, foi possível perceber um melhor desempenho do LeIA, seguido de Random Forest e Naive Bayes, apesar da última ter proporcionado melhor lucro. Levando em conta tanto acerto quanto lucro, todas as três técnicas performaram melhor, em longo prazo, do que quando nenhuma foi utilizada, ou seja, quando supôs-se simplesmente que o “gap” deveria ser fechado. Isso indica o sucesso da metodologia proposta e a possibilidade da utilização prática da ferramenta desenvolvida.

***Repositório contendo todos os códigos e bases de dados utilizados neste Trabalho de Conclusão de Curso, apresentado para obtenção do título de especialista em Data Science e Analytics (2024) do MBA USP/ESALQ.**
