# Qube data challenge 2022
The aim of this notebook is to present the main approaches
I have adopted to address the data challenge of Qube hosted by the Collège de France.
Here is the link: https://challengedata.ens.fr/challenges/72

The task is to predict the next returns from a stock market using a factor model. 
For a given stock, the factors are linear combinations of the past returns of this stock.

We  have  three-year track record  of 50 stocks and we have to provide (A, beta) as output. 
This output is then used to predict the returns of 50 other stocks over the same three-year period.

I have tried four type of models:
* Models using well-known factors
* Models based on linear regression
* Neural networks
* Models using the correlation over time.

The **best model** I have found is a **combination of a Ridge linear regression and some well known factors**,
which gives a train score of 0.1301 and a public score of 0.0829.

To participate in the challenge, I have used two accounts with pseudos *Line* and *Akina*.

For more details, see the jupyter notebook.