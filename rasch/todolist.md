1. read Rasch-PCM-MFRM-MFPCM.ipynb
2. make a jupyter notebook for each of the following topics. most of them are contained in Rasch-PCM-MFRM-MFPCM.ipynb
- every notebook must contain a simulation study of each of the five models
- Statistical inference is performed by Stan/cmdstanpy
- The identifiability condition and the convergence of NUTS algorithm need to be considered in creating the Stan model.
- MAP Estimation
    - this is a quick and fast method to obtain a point estimate
    - uncertainty computed by Laplace (Quadratic) approximation
    - draw a figure showing the scatter plot of (true, map) together with 96% credible interval
- Full Bayesian Inference by MCMC Sampling
- MCMC convergence analysis and apply PPC to verify the mcmc samples
- a scatter plot showing true_parameter versus computed posterior mean (and distribution (by boxplot)) for each of the parameter groups (e.g. person parameter group, item parmaeter group)
- draw a Wright map - a figure containing the histogram of person abilities and the locations of item parameters along the logit axis
- uncertainty analysis of each of the main parameters
- 96% credible interval
- the notebooks will be used as a teaching material for graduate students in Korean Language Education major and Global Korean Studies. Try to make explanations for them to accept the mathematical concept and computational results.
- Search for a recent research paper published after 2020 and provide the list at the end of the notebook; the search must include KCI Korean journals (한국에서 발간된 학술지 논문들을 포함하라)

3. Rasch Model
- 100 persons
- 20 items

4. MFRM Model
- 100 persons
- 4 items
    - at least one item must show a very well designed item characteristic (ordered step difficulty) as a prototype of item design
- 3 Likert scores [0, 1, 2] (2 step parameters)
- 5 rater severities
- After mcmc sampling, display the probability curves for each of the items
- Use the mean of the three difficulty parameters as the reprentation of the item like in the decomposed PCM model.

5. MFPCM Model
- the same as MFRM


6. Two PCM models are already created: IRT_P2_PCM_Combined.ipynb and IRT_P2_PCM_Decomposed.ipynb
- check the logical correctness of the contents in the two files
- modify if necessary

7. PCM with EIRT
- generated data samples have two item property (read and write) and three person properties (gender, country, school year(1,2,3))
- Stan model must use multi-level hierarchical model since there will be only a few students for a specific condition of gender, country, and school year.
- the number of students needs to be sufficiently large (> 100). Provide how to decide it practically and use the number in the simulation.

8. DIF with PCM
- Provide of a list of categories of DIF
- Explain real situations of DIF with PCM (together with practical examples)
- data file must contain a group id (for DIF simulation study)
- Stan code for DIF analysis with PCM model

9. Further Suggestion
- What will be the next step?
- Suggest a list of research topics (more than five) for master degree graduation study 
- The topics may be extended to include SEM

