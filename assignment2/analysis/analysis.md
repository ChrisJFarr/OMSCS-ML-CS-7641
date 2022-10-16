### Office Hours ### OH3: 1:45:50

mlrose->hivefork or hiive fork? It is actively maintained

Two sections

* Choose 3 problems and 4 algorithms
        * hill
        * ga
        * annealing
        * mimic
    * ga and mimic are more complicated
    * hill and annealing
    * requires structure like n-queens and k-color: ga or mimic handle structure
    * when doing results be sure to highlight specific algorithms (such as their strengths)
    track fitness score per iteration
    Very important topic: notion of convergence, set clear convergence criteria for algorithms
        * you could run them forever so we need a sensible stopping criteria, describe HOW it converged
    Try different problem sizes and see how convergence changes
        * I feel like I can create larger sizes for some of the problems and just have one size per problem
    Wall-clock time
    Function evaluations per iteration (some do more some do less)
        * which ones compute more than once per iteratino?
            * Genetic Algorithm: at each iteration, every animal is evaluated for fitness before 
                selecting the parents for the next generation. 

    Hypothesize why its interesting...
        * clear about how they are structured and how they will highlight a specific algorithm



* Use a randomized algorithm instead of backprop for neural-net
    * hill, ga, annealing
    * use ds1
    * use exact same structure...?
    * you may reimplement in mlrose (probably should do this, but target similar structure)
    * create the same graphs as done in assignment 1 for apples comparison
        * need to include convergence analysis using loss curve


######## End OH ####








Implement these randomized algorithms:
    randomized hill climbing
    simulated annealing
    a genetic algorithm
    MIMIC


You will then create (for sufficiently loose values of "create" including "steal", 
though it's fairly easy to come up with simple problems on your own in this case) 
three optimization problem domains

"optimization problem" is just a fitness function one is trying to maximize

Please note that the problems you create should be over discrete-valued parameter spaces. Bit strings are preferable.

You will apply **all four search techniques to these three** optimization problems. 12 total

* The first problem should highlight advantages of your genetic algorithm, 
* the second of simulated annealing, 
* and the third of MIMIC. Be creative and thoughtful. 
It is not required that the problems be complicated or painful. They can be simple. For example, the 4-peaks and 
k-color problems are rather straightforward, but illustrate relative strengths rather neatly.
(use 4-peaks, k-color, and one other)
my note: Perhaps try a few because I need to ensure a different algorithm does best for each (or similar pattern)


In addition to analyzing discrete optimization problems, you will also use the first three algorithms to 
find good weights for a neural network. In particular, you will use them instead of backprop for the neural 
network you used in assignment #1 on at least one of the problems you created for assignment #1. Notice 
that this assignment is about an optimization problem and about supervised learning problems. That probably 
means that looking at only the loss or only the accuracy won’t tell you the whole story. Luckily, you have 
already learned how to write an analysis on optimization problems and on supervised learning problems; now 
you just have to integrate your knowledge.

Apply randomized hill climbing, simulated annealing, and a genetic algorithm to the pytorch model...?


You must submit:

A file named README.txt that contains instructions for running your code and some way of getting to your code, just like last time.
a file named yourgtaccount-analysis.pdf that contains your writeup.
The file yourgtaccount-analysis.pdf should contain: 

the results you obtained running the algorithms on the networks: why did you get the results you did? what sort of changes might you make 
to each of those algorithms to improve performance? Feel free to include any supporting graphs or tables. And by "feel free to", of course, I mean "do".

a description of your optimization problems, and why you feel that they are interesting and exercise the strengths and weaknesses of each approach. Think hard about this.

analyses of your results. Beyond answering why you got the results you did you should compare and contrast the different algorithms. How fast 
were they in terms of wall clock time? Iterations? Which algorithm performed best? How do you define best? Be creative and think of as many questions 
you can, and as many answers as you can. You know the drill.

Note: Analysis writeup is limited to 10 pages total.

While looking at libraries, you might want to take a look at ABAGAIL:

Resources
https://github.com/pushkar/ABAGAIL
https://researchbank.swinburne.edu.au/file/ee09cd8d-64c2-402e-9388-b4c04ebcec30/1/PDF%20%286%20pages%29.pdf

python
https://mlrose.readthedocs.io/; https://mlrose.readthedocs.io/en/stable/source/algorithms.html


Notes for paper:

* discuss exploration vs exploitation
    * realize similarity between underfitting and overfitting


Analysis Notes

sa-general
* relationship between max-iters and max-max_attempts
    * max-attempts is the number of iterations the algorithm will continue to compute
        without any fitness score improvement. I made sure that this number was sufficiently
        large, but also needed to make sure that there were enough iterations to find
        a fall off in performance gains. For a larger problem the number of iterations must
        be larger and max-attempts should be slightly proportional to the number of iterations
        needed to find the optimal score.
* when computing, if the base parameters (those not part of the validation curve) were not
  set up well, then it wasn't clear if max_attempts was helpful. This is because the algorithm
  was stopping from the max_iters limit before it was able to find an optima. 
* 

sa-color

* performed validation curves on max_iters and max_attempts
* can see a sharp increase in performance as max_iters improves
* a quick increase in performance from max_attempts 0 to 100, levels off after. Found
  optimal performance with 300 or larger
* convergence: looking at fitness curve decided converged when it stopped
        improving for 5000 iterations. I believe that for this problem the 
        best that can be done is a score of 2 and SA finds this global optimum.
        I can't be sure, however, SA was able to solve very large problem sizes
        and I tuned the size until the best score was non-zero but still very small.
* Clock time per iter: 0.002259
* Iterations till best performance: 36,752
* Total time: 80.8544 seconds

sa-peaks

* performed validation curves on max_iters and max_attempts

* targeted convergence criteria: sa was able to achieve optimal scores for all problems (I believe). 
* A max-attempts that is proportional to max-iters ensured the algorithm could converge to the global optima for each problem.

mimic

hill-color
I see max-iters initially doen't appear to make a difference.

hill-peaks
* doesn't need a random restart, it is able to continue to improve and not get stuck
  in a local optima without restarts. able to find the global optima
* When there is structure there are more local optimas and the random-hill algorithm
  needs random restarts to try to find the global optima. However, in these tests
  random hill climbing failed to find the global optima.


ga general

When the problem gets an improvement from a small step, such as a single change 
to the state at any point, like the peaks problem, then ga does better with a smaller
mutation_prob, however if there are many local optima then a larger mutation_prob
is necessary to take larger steps to try and find an improved state. For all problems
the larger the population size the better it seems to do up to a point where Perhaps
having too large of population prevents moving in a direction and the lack of direction
causes it to get stuck more easily.

If not enough max-attempts or iterations are allowed then the algorithm doesn't have a 
chance to properly explore and will not converge.


Runtime Analysis

TODO Perform runtime analysis for each


ga-color
Clock time per iter: 1.597298
ga-queens
Clock time per iter: 4.416612
ga-peaks 
Clock time per iter: 0.069841

sa-color 
Clock time per iter: 0.000174
sa-queens
Clock time per iter: 0.000689
sa-peaks 
Clock time per iter: 0.000028

hill-color
Clock time per iter: 0.000101
hill-queens
Clock time per iter: 0.000515
hill-peaks 
Clock time per iter: 0.000024


mim-color  
Clock time per iter: 2.686540
mim-queens
Clock time per iter: 7.895370
mim-peaks 
Clock time per iter: 108.130921



Part 2

