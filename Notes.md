# Proj_SA

### Notas, Aula 30/04

Vamo-nos focar numa "Prova de Conceito" desta tecnologia.

(Se tivermos tempo seria interessante tentar abordar em que cenário este mecanismo poderia ter aplicação real)

Em vez de fazer uma app própria talvez fosse interessante fazer uma "API" ou um "Plug-In" que pode ser ajustado a diferentes aplicações.

Em vez de tentar identificar um utilizador específico talvez seja mais realista tentar detetar padrões, por exemplo "estudante de engenharia vs estudante de Letras" ou "jovens vs adultos (+)" ou "diferentes faixas etarias no computador familiar"

Um cenário que a professora mencionou que seria bastante interessante de trabalhar era no setor da saúde, para agilizar e tornar mais seguro o acesso aos computadores nos hospitais que são muitas vezes partilhados.

poderia ser interessante gerir acessos a determinados ficheiros, consoante o perfil traçado

Professora falou de mencionar LSTM's no trabalho futuro.

__________________________________________________________________________________________________________________________________

Sobre gamification secalhar sonvém dizer que pensamos num cenário em fizessemos um jogo para a parte da calibração, mas estar a colocar o utilizador em estado de competição poderia inviezar os dados

________________________________________________________

O Modelo do rato em particular, aprendeu a fazer "comparações", ou seja recebe um conjunto de métricas que estão associadas a um utilizador e ele é capaz de perceber se esses valores estão corretamente associados.

Como os dados dos nossos utilizadores virão principalmente do nosso uso, para criar exemplos de associação errada, utilizamos uma técnica de "Negative Sampling"/"Impostor Selection" onde associamos erradamente um sample a outro utilizador.
