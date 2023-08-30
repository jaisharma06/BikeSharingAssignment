# Bike Sharing Assignment
> A Multiple Linear Regression Assignment with RFE and MinMax Scaling.
___
## Table Of Contents
- [Bike Sharing Assignment](#bike-sharing-assignment)
  - [Table Of Contents](#table-of-contents)
  - [General Information](#general-information)
    - [Problem Statement](#problem-statement)
  - [Business Goal:](#business-goal)
  - [Technologies Used](#technologies-used)
  - [Conclusions](#conclusions)
    - [Final Equation](#final-equation)
    - [R-Squared Analysis](#r-squared-analysis)
    - [Suggestions](#suggestions)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)
    - [@jaisharma06](#jaisharma06)
___
## General Information
### Problem Statement
>A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short-term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.


>A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 


>In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to COVID-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.


>They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:

>Which variables are significant in predicting the demand for shared bikes?
>How well do those variables describe the bike demands
>Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors.
## Business Goal:
>You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market.

>For the given problem our target variable is 'cnt'. You have to find the factors affecting the count of total rental bikes including both casual and registered. 
> - Create a linear model that quantitatively relates rental bike count with the other variables.
> - Know the accuracy of the model.
___
## Technologies Used
>- Python - version 3.11.0
>- Matplotlib - version 3.7.2
>- Pandas - version 2.0.3
>- Seaborn - version 0.12.2
>- scikit-learn - 1.3.0
>- statsmodels - 0.14.0

## Conclusions
### Final Equation
**y = 0.0902 + (0.4914)Temperature + (0.2334)Year + (0.0970)Winter + (0.0916)September + (0.0645)Saturday + (0.0566)WorkingDay + (0.0527)Summer + (-0.0650)Spring + (-0.0786)Mist + Cloudy + (-0.3041)LightSnow**
### R-Squared Analysis
>R-Squared(Training Set) - 0.826 </br>
>Adj R-Squared(Training Set) - 0.822 </br>
>R-Squared(Testing Set) - 0.812 </br>
### Suggestions
>- Temperature could be a prime factor for making decisions for the Organisation
>- The demand for bikes was more in 2019 than in 2018.
>- Working days as they have a good influence on bike rentals. So it would be great to provide offers to the working individuals.
___
## Acknowledgements
>This project was made as an assignment for the Lending Club Case Study problem. Thank you UpGrad for providing us an opportunity to work on this.
___
## Contact
Created by -
### [@jaisharma06](https://github.com/jaisharma06)</br>
Name - Jai Prakash</br>
Email - [jai.sharma06@live.com](mailto:jai.sharma06@live.com)</br>
Phone - +91-8447490922

PDF - [Solved Subjective Questions](https://github.com/jaisharma06/BikeSharingAssignment/blob/main/linear_regression__subjective_questions_solved.pdf)

*Feel free to contact :smiley:!*
