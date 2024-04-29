# Econ 281: Computational Tools in Macroeconomics

### Instructors
Juan Herreño \
jherrenolopera@ucsd.edu \
Office Hours: Monday, 1:00pm-2:00pm (Atkinson Hall 6125)

Johannes Wieland \
jfwieland@ucsd.edu \
Office Hours: Thursday, 3:30pm-4:30pm (SDSC 197E)

### Course Description
Many modern macroeconomic models are be complicated. They feature dynamics, uncertainty, and market incompleteness. The objective of this class is to make you familiar with tools to solve this class of models. Much of the class will be hands-on, like a "lab." We will only scratch the surface of computational methods that serve as a stepping stone for solving more specialized models you might encounter in the future.

### Software

We will be using both Matlab, Python, and Jupyter notebooks. Make sure they working and up-to-date on your personal computer before the first class. We want to avoid wasting time installing and troubleshooting software.
1. Matlab: [get it from UCSD](https://blink.ucsd.edu/technology/computers/software-acms/available-software/matlab.html)
2. Python and Jupyter notebooks: [use Anaconda distribution](https://www.anaconda.com/download/)  
For effective use of Python and Jupyter notebooks we suggest an editor such as Sublime or Visual Studio Code. 

We use both languages because certain packages that are helpful for solving macro models are only written in one of them. You are welcome to use other languages as well, but we will be less familiar with them. 

### Homework 

We will assign computational homework every week. You should store all of your homework in a single GitHub repository. You should give us access to your repository so we can pull the code and run it on our computers. For submissions in Python we recommend using Julia notebooks. 

### Class Project 

The class project can be either your own work or a replication. 
1. Own work: Should solve an interesting model using the computational tools learned in class. What makes a model "interesting" is generally matching a moment of the data.
2. Replication: The paper should be unpublished and the code should not publicly available. Please send us your replication choice by the end of week 2 for approval.

Your own work can also be combined with the class project for Econ 281: Micro to Macro. In this case we expect that your submission contains an important computational component using the methods in this class as well as a micro-to-macro component. But the components need not be split evenly.

### Grading

The first iteration for the class project will count for 10% of the grade. The presentation will count for 10% of the grade. The final draft will count for 30% of the grade. Homework will count for 50%.

### Auditing

We expect students who audit the class to participate and do the homework. If you want to submit your class project and get feedback on it, you need to take the class for grade.

## Topics

1. **Let's get started on the computer** (JW)

    Dynamic programming and why it works. Value function iteration, functional approximation, interpolation.

    *Homework*: Cake-eating problem with and without depreciation. Finite vs infinite horizon. Basic consumption problem.

2. **Policy function iteration** (JW)

    Solver (hill-climbing), convergence criteria. Curse of dimensionality (either here or 1 or 3)

    *Homework*: Compare policy functions and value functions in cake eating and basic consumption problem.

3. **Continuous time** (JH)

    Continuous time limit, HJB equation (solve giving a price). Simulate given a price. 

    *Homework*: Compare your solution with discrete time. Repeat the exercise of the lab using the implicit method. Replace an endowment with profits from a production function. Choose your production technology.

4. **Integration and EGM** (JH)

    Introduce risk, Expectations. Simulate given a price. Targeting moments.

    *Homework*: Simulate starting from policy functions with idiosyncratic risk. Target "complicated" moments. Compute aggregate demand for an asset.

5. **Bewley and Aiyagari Models** (JH)

    How to clear a market and put everything together. Shift aggregate supply and demand.

    *Homework*: Solve a model with heterogeneous firms and adjustment costs.

6. **Reiter method** (JH)

    Include aggregate disturbances to a state-space representation problem

    *Homework*: Solve Aiyagari with aggregate shocks

7. **Sequence Space	with Representative Agent** (JW)

    NK model, DAGs

    *Homework*: TANK + TANK and RA equivalence


8. **Sequence Space	with Heterogeneous Agents** (JW)

    Jacobians in HANK

    *Homework*: SHADE models

9. **Applications** (JH/JW)

    TBD

10. **Student Presentations**

    TBD    
