instruction_maths_prm = "Please convert natural language plans into a series of subgoals and their corresponding actions" \
                "that lead to the successful implementation with respect to the given instructions. " \
                "Please use 'R[number]' to represent the intermediate results for each subgoal, without generating" \
                "any exact values. Please also use functions to represent the corresponding actions. " \
                "For the actions, they must be one of 'Calculator', 'SetEquation', 'SolveEquation', 'Count', 'SolveInequality', 'Code', and 'Define'.\n\n" \
                "Example 1:\n" \
                "Instruction: The first four terms in an arithmetic sequence are $x+y$, $x-y$, $xy$, and $x/y$, in that order. What is the fifth term? Express your answer as a common fraction.\n\n" \
                "Natural language plan:\n" \
                "Since the difference of the first two terms is $-2y$, the third and fourth terms of the sequence must be $x-3y$ and $x-5y$. Thus \\[\nx-3y = xy \\quad\\text{and}\\quad x-5y = \\frac{x}{y},\n\\]so $xy - 5y^{2} = x.$ Combining these equations we obtain \\[\n(x - 3y) - 5y^{2}= x\\quad\\text{and, therefore, }\\quad -3y - 5y^{2} = 0.\n\\]Since $y$ cannot be 0, we have $y = -\\frac{3}{5}$, and it follows that $x = -\\frac{9}{8}$. The fifth term in the sequence is $x - 7y\n= \\boxed{\\frac{123}{40}}$.\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Calculate the difference between two consecutive terms in the sequence.\n" \
                "Action 1-1: R1 = Calculator((x-y) - (x+y)) = -2y \n\n" \
                "Subgoal 2: Calculate the third, fourth and fifth terms according to the calculated difference.\n" \
                "Action 2-1: R2 = Calculator((x+y) + 2*R1) = x-3y \n" \
                "Action 2-2: R3 = Calculator((x+y) + 3*R1) = x-5y \n" \
                "Action 2-3: R4 = Calculator((x+y) + 4*R1) = x-7y \n\n" \
                "Subgoal 3: Solve x and y by solving equations.\n" \
                "Action 3-1: R5, R6 = SolveEquation(R2=xy, R3=x/y) = -9/8, -3/5 \n\n" \
                "Subgoal 4: Calculate the real value of the fifth term using x and y values.\n" \
                "Action 4-1: Output = Calculator(R4(R5, R6)) = 123/40 \n\n" \
                "Example 2:\n" \
                "Task: How many positive integers less than or equal to 100 have a prime factor that is greater than 4?\n\n" \
                "Natural language plan: The easiest solution is to find the number of positive integers with only 2 and 3 as their prime factors. If the number has no factors of 3, the qualifying numbers are $2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6$ for 7 total. If there is one factor of 3, we have $2^0 \cdot 3^1, 2^1 \cdot 3^1, 2^2 \cdot 3^1, 2^3 \cdot 3^1, 2^4 \cdot 3^1, 2^5 \cdot 3^1$ for 6 total. With two factors of 3, we have $2^0 \cdot 3^2, 2^1 \cdot 3^2, 2^2 \cdot 3^2, 2^3 \cdot 3^2$ for 4 total. With three factors of 3, we have $2^0 \cdot 3^3, 2^1 \cdot 3^3$ for 2 total. Finally, $3^4$ gives us 1 more. So, there are $7+ 6+4+2+1 = 20$ positive integers less than or equal to 100 that have only 2 and 3 as prime factors. Therefore, there are $100-20 = \boxed{80}$ positive integers less than or equal to 100 that have a prime factor greater than 4.\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Calculate the number of positive integers with only 2 as their prime factor.\n" \
                "Action 1-1: R1 = Count(2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6) = 7 \n\n" \
                "Subgoal 2: Calculate the number of positive integers with 2 and 3 as their prime factors, with one factor of 3.\n" \
                "Action 2-1: R2 = Count(2^0 * 3^1, 2^1 * 3^1, 2^2 * 3^1, 2^3 * 3^1, 2^4 * 3^1, 2^5 * 3^1) = 6 \n\n" \
                "Subgoal 3: Calculate the number of positive integers with 2 and 3 as their prime factors, with two factors of 3.\n" \
                "Action 3-1: R3 = Count(2^0 * 3^2, 2^1 * 3^2, 2^2 * 3^2, 2^3 * 3^2) = 4 \n\n" \
                "Subgoal 4: Calculate the number of positive integers with 2 and 3 as their prime factors, with three factors of 3.\n" \
                "Action 4-1: R4 = Count(2^0 * 3^3, 2^1 * 3^3) = 2 \n\n" \
                "Subgoal 5: Calculate the number of positive integers with 3 as their prime factor, with four factors of 3.\n" \
                "Action 5-1: R5 = Count(3^4) = 1 \n\n" \
                "Subgoal 6: Calculate the total number of positive integers less than or equal to 100 that have only 2 and 3 as prime factors.\n" \
                "Action 6-1: R6 = Calculator(R1 + R2 + R3 + R4 + R5) = 20 \n\n" \
                "Subgoal 7: Calculate the number of positive integers less than or equal to 100 that have a prime factor greater than 4.\n" \
                "Action 7-1: Output = Calculator(100 - R6) = 80 \n\n" \
                "Example 3:\n" \
                "The sum of the squares of three consecutive positive even numbers is $12296$. Find the product of the three numbers divided by $8$.\n\n" \
                "Natural language plan: If $n$ is the middle number of the three, the other two numbers are $n-2$ and $n+2$. Therefore, the squares are $n^2-4n+4$, $n^2$, and $n^2+4n+4$. Setting the sum of the three squares equal to $12296$. Because $n$ is positive, $n$ must be $64$. Therefore, the set of numbers is $62, 64, 66$. The product of those is $261888$. The product divided by 8 is $\boxed{32736}$.\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Define the three consecutive even numbers in terms of n.\n" \
                "Action 1-1: R1 = Define(n-2) = n-2 \n" \
                "Action 1-2: R2 = Define(n) = n \n" \
                "Action 1-3: R3 = Define(n+2) = n+2 \n\n" \
                "Subgoal 2: Calculate the squares of the three numbers.\n" \
                "Action 2-1: R4 = Calculator(R1^2) = n^2-4n+4 \n" \
                "Action 2-2: R5 = Calculator(R2^2) = n^2 \n" \
                "Action 2-3: R6 = Calculator(R3^2) = n^2+4n+4 \n\n" \
                "Subgoal 3: Set up and solve the equation for the sum of the squares.\n" \
                "Action 3-1: R8 = SolveEquation(R4 + R5 + R6 = 12296) = 64 \n\n" \
                "Subgoal 4: Determine the positive solution for n.\n" \
                "Action 4-1: R9 = Code(If R8 > 0, R8, -R8) = 64 \n\n" \
                "Subgoal 5: Calculate the product of the three numbers.\n" \
                "Action 5-1: R10 = Calculator(R1(R9)) = 62 \n" \
                "Action 5-2: R11 = Calculator(R2(R9)) = 64 \n" \
                "Action 5-3: R12 = Calculator(R3(R9)) = 66 \n" \
                "Action 5-4: R13 = Calculator(R10 * R11 * R12) = 261888 \n\n" \
                "Subgoal 6: Divide the product by 8 to get the final answer.\n" \
                "Action 6-1: Output = Calculator(R13 / 8) = 32736 \n\n" \
                "Example 4:\n" \
                "For how many different digits $n$ is the three-digit number $14n$ divisible by $n$? Note: $14n$ refers to a three-digit number with the unit digit of $n,$ not the product of $14$ and $n.$\n\n" \
                "Natural language plan: We have to account for each possible value of $n$ here. First of all, we can quickly find that for $n = 1, 2, 5,$ the resulting number $14n$ must be divisible by $n$, using their respective divisibility rules.\n" \
                "We see that for $n = 3$, we get $143.$ Since $1 + 4 + 3 = 8,$ which is not a multiple of $3,$ we can see that $n = 3$ does not work. Moreover, if $143$ is not divisible by $3$, then $146$ and $149$ are not divisible by $3$ or any multiple of $3$, so $n = 6$ and $n = 9$ do not work.\n" \
                "For $n = 4$, we can see that $144$ is divisible by $4$ because $44$ is divisible by $4,$ so $n = 4$ works. For $n = 7$, we can easily perform division and see that $147$ is divisible by $7,$ so $n = 7$ works. For $n = 8$, we have little choice but to find that $\dfrac{148}{8} = \dfrac{37}{2},$ and so $n = 8$ does not work. All in all, we have that $n$ can be $1,$ $2,$ $4,$ $5,$ or $7,$ so we have $\boxed{5}$ possible choices for $n$ such that $14n$ is divisible by $n.$\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Check if the three-digit number 14n is divisible by n for n = 1, 2, 5.\n" \
                "Action 1-1: R1 = Code(If 141 % 1 == 0, 1, 0) = 1 \n" \
                "Action 1-2: R2 = Code(If 142 % 2 == 0, 1, 0) = 1 \n" \
                "Action 1-3: R3 = Code(If 145 % 5 == 0, 1, 0) = 1 \n\n" \
                "Subgoal 2: Check if the three-digit number 14n is divisible by n for n = 3, 6, 9.\n" \
                "Action 2-1: R4 = Code(If 143 % 3 == 0, 1, 0) = 0 \n" \
                "Action 2-2: R5 = Code(If 146 % 6 == 0, 1, 0) = 0 \n" \
                "Action 2-3: R6 = Code(If 149 % 9 == 0, 1, 0) = 0 \n\n" \
                "Subgoal 3: Check if the three-digit number 14n is divisible by n for n = 4, 7, 8.\n" \
                "Action 3-1: R7 = Code(If 144 % 4 == 0, 1, 0) = 1 \n" \
                "Action 3-2: R8 = Code(If 147 % 7 == 0, 1, 0) = 1 \n" \
                "Action 3-3: R9 = Code(If 148 % 8 == 0, 1, 0) = 0 \n\n" \
                "Subgoal 4: Calculate the total number of different digits n for which the three-digit number 14n is divisible by n.\n" \
                "Action 4-1: Output = Calculator(R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9) = 5 \n\n" \
                "Now please help us generate a plan consisting of subgoals according to the following instruction and its natural language plan!\n\n"


instruction_maths_gsm = "Please convert natural language plans into a series of subgoals and their corresponding actions" \
                "that lead to the successful implementation with respect to the given instructions. " \
                "Please use 'R[number]' to represent the intermediate results for each subgoal, without generating" \
                "any exact values. Please also use functions to represent the corresponding actions. " \
                "For the actions, they must be one of 'Calculator', 'SolveEquation', 'Count', and 'Define'.\n\n" \
                "Example 1:\n" \
                "Task: Peter goes to the store to buy a soda. The soda costs $.25 an ounch. He brought $2 with him and leaves with $.50. How many ounces of soda did he buy?\n\n" \
                "Natural language plan:\n" \
                "He spend $1.5 on soda because 2 - .5 = <<2-.5=1.5>>1.5 He bought 6 ounces of soda because 1.5 / .25 = <<6=6>>6\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Calculate how much the soda costs in total.\n" \
                "Action 1-1: R1 = Calculator(2 - 0.5) = 1.5 \n\n" \
                "Subgoal 2: Calculate the ounces of soda the price per ounch.\n" \
                "Action 2-1: R2 = Calculator(R1 / 0.25) = 6 \n\n" \
                "Example 2:\n" \
                "Task: Leah earned $28 working odd jobs around the neighborhood. She spent a seventh of it on a milkshake and put half of the rest in her savings account. She left the remaining money in her wallet. Her dog got ahold of her wallet and shredded all the money inside but $1. How many dollars did Leah lose?\n\n" \
                "Natural language plan: Leah spent 28 * 1/7 = $<<28/7=4>>4 on a milkshake. She had 28 - 4 = $<<28-4=24>>24 left. She put half in her savings account and half in her wallet, so she had 24 / 2 = $<<24/2=12>>12 in her wallet. Her dog shredded all the money in her wallet but $1, so Leah lost 12 - 1 = $<<12-1=11>>11.\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Calculate the amount Leah spent on milkshake.\n" \
                "Action 1-1: R1 = Calculator(28 * 1/7) = 4 \n\n" \
                "Subgoal 2: Calculate Leah's rest money after buying milkshake.\n" \
                "Action 2-1: R2 = Calculator(28 - R1) = 24 \n\n" \
                "Subgoal 3: Calculate the amount Leah spent in her savings account.\n" \
                "Action 3-1: R3 = Calculator(R2 / 2) = 12 \n\n" \
                "Subgoal 4: Calculate the amount Leah remained in her wallet.\n" \
                "Action 4-1: R4 = Calculator(R2 - R3) = 12 \n\n" \
                "Subgoal 5: Calculate the amount Leah lost after her dog shredded all the money in her wallet.\n" \
                "Action 5-1: R5 = Calculator(R4 - 1) = 11 \n\n" \
                "Example 3:\n" \
                "Task: Mrs. Snyder used to spend 40% of her monthly income on rent and utilities. Her salary was recently increased by $600 so now her rent and utilities only amount to 25% of her monthly income. How much was her previous monthly income?\n\n" \
                "Natural language plan:\n" \
                "Let her previous monthly income be p The cost of her rent and utilities was 40% of p which is (40/100)*p = 2p/5 Her income was increased by $600 so it is now p+$600 The cost of her rent and utilities now amount to 25% of (p+$600) which is (25/100)*(p+$600) = (p+$600)/4 Equating both expressions for cost of rent and utilities: 2p/5 = (p+$600)/4 Multiplying both sides of the equation by 20 gives 8p = 5p+$3000 Subtracting 5p from both sides gives: 3p = $3000 Dividing both sides by 3 gives p = $1000\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Define Snyder's previous monthly income be p.\n" \
                "Action 1-1: R1 = Define(p) = p \n\n" \
                "Subgoal 2: Calculate the amount that Mrs. Snyder used to spend for rent and utilities in terms of p.\n" \
                "Action 2-1: R2 = Calculator(40% * p) = 2p/5 \n\n" \
                "Subgoal 3: Calculate the increased salary.\n" \
                "Action 3-1: R3 = Calculator(p + 600) = p + 600 \n\n" \
                "Subgoal 4: Calculate the amount that Mrs. Snyder used to spend for rent and utilities based on the increased salary.\n" \
                "Action 4-1: R4 = Calculator(25% * R3) = (p + 600)/4 \n\n" \
                "Subgoal 5: Setup equations and solve p.\n" \
                "Action 5-1: R5 = SolveEquation(R2 = R4) = 1000 \n\n" \
                "Example 4:\n" \
                "Ralph is going to practice playing tennis with a tennis ball machine that shoots out tennis balls for Ralph to hit. He loads up the machine with 175 tennis balls to start with. Out of the first 100 balls, he manages to hit 2/5 of them. Of the next 75 tennis balls, he manages to hit 1/3 of them. Out of all the tennis balls, how many did Ralph not hit?\n\n" \
                "Natural language plan: Out of the first 100 balls, Ralph was able to hit 2/5 of them and not able to hit 3/5 of them, 3/5 x 100 = 60 tennis balls Ralph didn't hit. Out of the next 75 balls, Ralph was able to hit 1/3 of them and not able to hit 2/3 of them, 2/3 x 75 = 50 tennis balls that Ralph didn't hit. Combined, Ralph was not able to hit 60 + 50 = <<60+50=110>>110 tennis balls Ralph didn't hit.\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Calculate the tennis balls Ralph didn't hit for the first 100 balls.\n" \
                "Action 1-1: R1 = Calculator(1 - 2/5) = 3/5 \n" \
                "Action 1-2: R2 = Calculator(100 * R1) = 60 \n\n" \
                "Subgoal 2: Calculate the tennis balls Ralph didn't hit for the next 75 balls.\n" \
                "Action 2-1: R3 = Calculator(1 - 1/3) = 2/3 \n" \
                "Action 2-2: R4 = Calculator(75 * R3) = 50 \n\n" \
                "Subgoal 3: Calculate the total number of the tennis balls Ralph didn't hit.\n" \
                "Action 3-1: R5 = SolveEquation(R2 + R4) = 110 \n\n" \
                "Example 5:\n" \
                "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?\n\n" \
                "Natural language plan: She works 8 hours a day for $18 per hour so she makes 8*18 = $<<8*18=144.00>>144.00 per 8-hour shift She works 10 hours a day and anything over 8 hours is eligible for overtime, so she gets 10-8 = <<10-8=2>>2 hours of overtime Overtime is calculated as time and a half so and she makes $18/hour so her overtime pay is 18*.5 = $<<18*.5=9.00>>9.00 Her overtime pay is 18+9 = $<<18+9=27.00>>27.00 Her base pay is $144.00 per 8-hour shift and she works 5 days and makes 5 * $144 = $<<144*5=720.00>>720.00 Her overtime pay is $27.00 per hour and she works 2 hours of overtime per day and makes 27*2 = $<<27*2=54.00>>54.00 in overtime pay 2 hours of overtime pay for 5 days means she makes 54*5 = $270.00 In 5 days her base pay is $720.00 and she makes $270.00 in overtime pay so she makes $720 + $270 = $<<720+270=990.00>>990.00\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Calculate the wage if she works 8 hours a day for $18 per hour.\n" \
                "Action 1-1: R1 = Calculator(8 * 18) = 144 \n\n" \
                "Subgoal 2: Calculate the amount of overtime if she works 10 hours.\n" \
                "Action 2-1: R2 = Calculator(10 - 8) = 2 \n\n" \
                "Subgoal 3: Calculate the overtime pay per hour.\n" \
                "Action 3-1: R3 = Calculator(18 + 1/2 * 18) = 27 \n\n" \
                "Subgoal 4: Calculate the overtime pay per day.\n" \
                "Action 4-1: R4 = Calculator(R2 * R3) = 54 \n\n" \
                "Subgoal 5: Calculate the base pay for 5 days.\n" \
                "Action 5-1: R5 = Calculator(R1 * 5) = 720 \n\n" \
                "Subgoal 6: Calculate the overtime pay for 5 days.\n" \
                "Action 6-1: R6 = Calculator(R4 * 5) = 270 \n\n" \
                "Subgoal 7: Calculate the total pay for 5 days.\n" \
                "Action 7-1: R7 = Calculator(R5 + R6) = 990 \n\n" \
                "Example 6:\n" \
                "Toby is counting goldfish in the local pond. He knows that only 25% of goldfish are at the surface and the rest are too deep below the surface to be able to see. If he counts 15 goldfish, how many are below the surface?\n\n" \
                "Natural language plan: There are 60 goldfish because 15 / .25 = <<15/.25=60>>60 There are 45 goldfish below the surface because 60 x (1-0.25) = <<60*(1-0.25)=45>>45\n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Calculate the total number of goldfish in the local pond.\n" \
                "Action 1-1: R1 = Calculator(15 / 25%) = 60 \n\n" \
                "Subgoal 2: Calculate the ratio of goldfish below the surface.\n" \
                "Action 2-1: R2 = Calculator(1 - 25%) = 0.75 \n\n" \
                "Subgoal 3: Calculate the number of goldfish below the surface.\n" \
                "Action 3-1: R3 = Calculator(R1 * R2) = 45 \n\n" \
                "Now please help us generate a plan consisting of subgoals according to the following instruction and its natural language plan!\n\n"


instruction_qa = "Please convert natural language plans into a series of subgoals and their corresponding actions" \
                "that lead to the successful implementation with respect to the given instructions. " \
                "Please use 'R[number]' to represent the intermediate results for each subgoal, without generating" \
                "any exact values. Please also use functions to represent the corresponding actions. " \
                "For the actions, they must be one of 'KnowledgeQuery', 'ParagraphRetrieve', 'QA', 'Calculator' and 'Code'.\n\n" \
                "Example 1:\n" \
                "Task: Are more people today related to Genghis Khan than Julius Caesar?\n\n" \
                "Natural language plan:\n" \
                "We find relevant facts: Julius Caesar had three children. Genghis Khan had sixteen children. Modern geneticists have determined that out of every 200 men today has DNA that can be traced to Genghis Khan. We need to answer these questions: 1. How many kids did Julius Caesar have? (Can be answered based on paragraph 'Julius Caesar-75') 2. How many kids did Genghis Khan have? (Can be answered based on paragraph 'Genghis Khan-17') 3. Is #2 greater than #1? Based on these evidences and decomposed questions, the answer is True. \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Obtain the number of the kids that Julius Caesar had.\n" \
                "Action 1-1: R1 = KnowledgeQuery(Julius Caesar) = WikipediaPage(Julius Caesar). \n" \
                "Action 1-2: R2 = ParagraphRetrieve(R1, Query: How many kids did Julius Caesar have?) = Paragraph(Julius Caesar-75). \n" \
                "Action 1-3: R3 = QA([R2], Question: How many kids did Julius Caesar have?) = 3. \n\n" \
                "Subgoal 2: Obtain the number of the kids that Genghis Khan had.\n" \
                "Action 2-1: R4 = KnowledgeQuery(Genghis Khan) = WikipediaPage(Genghis Khan). \n" \
                "Action 2-2: R5 = ParagraphRetrieve(R4, Query: How many kids did Genghis Khan have?) = Paragraph(Genghis Khan-17). \n" \
                "Action 2-3: R6 = QA([R5], Question: How many kids did Genghis Khan have?) = 16. \n\n" \
                "Subgoal 3: Determine if Genghis Khan had more kids.\n" \
                "Action 3-1: R7 = Calculator(R6 > R3) = True \n\n" \
                "Example 2:\n" \
                "Task: Would a Monoamine Oxidase candy bar cheer up a depressed friend?\n\n" \
                "Natural language plan:\n" \
                "We find relevant facts: Depression is caused by low levels of serotonin, dopamine and norepinephrine. Monoamine Oxidase breaks down neurotransmitters and lowers levels of serotonin, dopamine and norepinephrine. We need to answer these questions: 1. Depression is caused by low levels of what chemicals? (Can be answered based on paragraph 'Depression (mood)-13') 2. Monoamine Oxidase has an effect on what chemicals? (Can be answered based on paragraph 'Monoamine oxidase-8') 3. Of the chemicals listed in both #1 and #2, does Monoamine Oxidase raise their levels? (Can be answered based on paragraph 'Serotonin-36') Based on these evidences and decomposed questions, the answer is False. \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Answer low levels of what chemicals cause depression.\n" \
                "Action 1-1: R1 = KnowledgeQuery(depression) = WikipediaPage(Depression (mood)). \n" \
                "Action 1-2: R2 = ParagraphRetrieve(R1, Query: Depression is caused by low levels of what chemicals?) = Paragraph(Depression (mood)-13). \n" \
                "Action 1-3: R3 = QA([R2], Question: Depression is caused by low levels of what chemicals?) = serotonin, dopamine and norepinephrine. \n\n" \
                "Subgoal 2: Answer what chemicals Monoamine Oxidase has an effect on.\n" \
                "Action 2-1: R4 = KnowledgeQuery(Monoamine Oxidase) = WikipediaPage(Monoamine oxidase). \n" \
                "Action 2-2: R5 = ParagraphRetrieve(R4, Query: Monoamine Oxidase has an effect on what chemicals?) = Paragraph(Monoamine oxidase-8). \n" \
                "Action 2-3: R6 = QA([R5], Question: Monoamine Oxidase has an effect on what chemicals?) = serotonin, dopamine and norepinephrine. \n\n" \
                "Subgoal 3: Determine if Monoamine Oxidase raise the levels of these chemicals. \n" \
                "Action 3-1: R7 = QA([R3, R6], Question: Does Monoamine Oxidase raise the levels of these chemicals?) = False. \n\n" \
                "Example 3:\n" \
                "Task: Will the Albany in Georgia reach a hundred thousand occupants before the one in New York?\n\n" \
                "Natural language plan:\n" \
                "We find relevant facts: Albany, GA has around 75,000 people Albany, NY has almost 100,000 people We need to answer these questions: 1. What is the population of Albany, Georgia? (Can be answered based on paragraph 'Albany, Georgia-1') 2. What is the population of Albany, New York? (Can be answered based on paragraph 'Albany, New York-2') 3. What is the difference between 100,000 and #1? 4. What is the difference between 100,000 and #2? 5. Is #3 smaller than #4? Based on these evidences and decomposed questions, the answer is False. \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Obtain the population of Albany, Georgia.\n" \
                "Action 1-1: R1 = KnowledgeQuery(Albany, Georgia) = WikipediaPage(Albany, Georgia). \n" \
                "Action 1-2: R2 = ParagraphRetrieve(R1, Query: What is the population of Albany, Georgia?) = Paragraph(Albany, Georgia-1). \n" \
                "Action 1-3: R3 = QA([R2], Question: What is the population of Albany, Georgia?) = 75000. \n\n" \
                "Subgoal 2: Obtain the population of Albany, New York.\n" \
                "Action 2-1: R4 = KnowledgeQuery(Albany, New York) = WikipediaPage(Albany, New York). \n" \
                "Action 2-2: R5 = ParagraphRetrieve(R4, Query: What is the population of Albany, New York?) = Paragraph(Albany, New York-2). \n" \
                "Action 2-3: R6 = QA([R5], Question: What is the population of Albany, New York?) = 100000. \n\n" \
                "Subgoal 3: Calculate the difference between 100,000 and the population of Albany, Georgia. \n" \
                "Action 3-1: R7 = Calculator(100000 - R3) = 25000. \n\n" \
                "Subgoal 4: Calculate the difference between 100,000 and the population of Albany, New York. \n" \
                "Action 4-1: R8 = Calculator(100000 - R6) = 0. \n\n" \
                "Subgoal 5: Determine if the difference calculated in Subgoal 3 is smaller than the one calculated in Subgoal 4. \n" \
                "Action 5-1: R9 = Calculator(R7 < R8) = False. \n\n" \
                "Example 4:\n" \
                "Task: Would a dog respond to bell before Grey seal?\n\n" \
                "Natural language plan:\n" \
                "We find relevant facts: Grey seals have no ear flaps and their ears canals are filled with wax. Grey seals hear better underwater when their ears open like a valve. Dogs have sensitive ears that can hear as far as a quarter of a mile away. We need to answer these questions: 1. How sensitive is a grey seal's hearing on land? (Can be answered based on paragraph 'Pinniped-24') 2. How sensitive is a dog's hearing on land? (Can be answered based on paragraph 'Hearing range-11', 'Hertz-5') 3. Is #2 better than #1? Based on these evidences and decomposed questions, the answer is True. \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Answer how sensitive a grey seal's hearing on land is.\n" \
                "Action 1-1: R1 = KnowledgeQuery(grey seal's hearing) = WikipediaPage(Pinniped). \n" \
                "Action 1-2: R2 = ParagraphRetrieve(R1, Query: How sensitive is a grey seal's hearing on land?) = Paragraph(Pinniped-24). \n" \
                "Action 1-3: R3 = QA([R2], Question: How sensitive is a grey seal's hearing on land?) = Grey seals have no ear flaps and their ears canals are filled with wax. Grey seals hear better underwater when their ears open like a valve. \n\n" \
                "Subgoal 2: Answer how sensitive a dog's hearing on land is.\n" \
                "Action 2-1: R4 = KnowledgeQuery(dog's hearing) = WikipediaPage(Hearing range), WikipediaPage(Hertz). \n" \
                "Action 2-2: R5 = ParagraphRetrieve(R4, Query: How sensitive is a dog's hearing on land?) = Paragraph(Hearing range-11), Paragraph(Hertz-5). \n" \
                "Action 2-3: R6 = QA([R5], Question: How sensitive is a dog's hearing on land?) = Dogs have sensitive ears that can hear as far as a quarter of a mile away. \n\n" \
                "Subgoal 3: Determine if dog's hearing is better than Grey seal's. \n" \
                "Action 3-1: R7 = QA([R3, R6], Question: Is dog's hearing better than Grey seal's?) = True. \n\n" \
                "Example 5:\n" \
                "Task: Could the members of The Police perform lawful arrests?\n\n" \
                "Natural language plan:\n" \
                "We find relevant facts: The members of The Police were musicians, not law enforcement officers. Only law enforcement officers can perform lawful arrests. We need to answer these questions: 1. Who can perform lawful arrests? (Can be answered based on paragraph 'Arrest-2') 2. Are members of The Police also #1? (Can be answered based on paragraph 'Citizen's arrest-2', 'The Police-1') Based on these evidences and decomposed questions, the answer is False. \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Answer who can perform lawful arrests.\n" \
                "Action 1-1: R1 = KnowledgeQuery(lawful arrests) = WikipediaPage(Arrest). \n" \
                "Action 1-2: R2 = ParagraphRetrieve(R1, Query: Who can perform lawful arrests?) = Paragraph(Arrest-2). \n" \
                "Action 1-3: R3 = QA([R2], Question: Who can perform lawful arrests?) = law enforcement officers. \n\n" \
                "Subgoal 2: Answer if members of The Police are also law enforcement officers.\n" \
                "Action 2-1: R4 = KnowledgeQuery(The Police) = WikipediaPage(The Police), WikipediaPage(Citizen's arrest). \n" \
                "Action 2-2: R5 = ParagraphRetrieve(R4, Query: Are members of The Police also law enforcement officers?) = Paragraph(Citizen's arrest-2), Paragraph(The Police-1). \n" \
                "Action 2-3: R6 = QA([R5], Question: Are members of The Police also law enforcement officers?) = False. \n\n" \
                "Subgoal 3: Determine if the members of The Police can perform lawful arrests.\n" \
                "Action 3-1: R7 = QA([R3, R6], Question: Can the members of The Police perform lawful arrests) = False. \n\n" \
                "Now please help us generate a plan consisting of subgoals according to the following instruction and its natural language plan! \n\n"


instruction_web_agent = "Please convert natural language plans into a series of subgoals and their corresponding actions" \
                "that lead to the successful implementation with respect to the given instructions. " \
                "Please use 'R[number]' to represent the intermediate results for each subgoal, without generating" \
                "any exact values. Please also use functions to represent the corresponding actions. " \
                "For the actions, they must be one of 'TYPE', 'CLICK', and 'SELECT'.\n\n" \
                "Example 1:\n" \
                "Task: Book a flight + cruise flying from Atlanta in October for vacation starting on October 8 for 6 nights with Miami as the departure port, choose the cheapest flight, hotel, and room in the cruise for booking.\n\n" \
                "Natural language plan:\n" \
                "[span]  Flights + Cruise -> CLICK; [button]  Search flights + cruise External Link should open ... -> CLICK; [combobox]  Departing from -> TYPE: ATLANTA; [span]  Atlanta, GA (ATL) -> CLICK; [span]  Jun 2023 -> CLICK; [label]  October 08, 2023 -> CLICK; [span]  6 Nights -> SELECT; [label]  Miami -> CLICK; [button]  View details -> CLICK; [link]  Select package -> CLICK; [span]  Checkout -> CLICK \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Enter the site for flight + cruise.\n" \
                "Action 1-1: R1 = CLICK(Env, QUERY: Enter the site for flight + cruise). \n\n" \
                "Subgoal 2: Open the link for searching flights and cruise.\n" \
                "Action 2-1: R2 = CLICK(R1, QUERY: Open the link for searching flights and cruise). \n\n" \
                "Subgoal 3: Type Atlanta as the departure location.\n" \
                "Action 3-1: R3 = TYPE(R2, QUERY: Type Atlanta as the departure location, TEXT: Atlanta). \n\n" \
                "Subgoal 4: Choose the option Atlanta, GA (ATL) to fill the departure location.\n" \
                "Action 4-1: R4 = CLICK(R3, QUERY: Choose the option Atlanta, GA (ATL)). \n\n" \
                "Subgoal 5: Click the date region to set the departure date.\n" \
                "Action 5-1: R5 = CLICK(R4, QUERY: Click the date region to set the departure date). \n\n" \
                "Subgoal 6: Choose October 2023 as the departure date.\n" \
                "Action 6-1: R6 = CLICK(R5, QUERY: Choose October 08, 2023 as the departure date). \n\n" \
                "Subgoal 7: Set 6 nights as the living period for the hotel.\n" \
                "Action 7-1: R7 = SELECT(R6, QUERY: Click 6 nights as the living period for the hotel, TEXT: 6 nights). \n\n" \
                "Subgoal 8: Choose Miami as the destination.\n" \
                "Action 8-1: R8 = CLICK(R7, QUERY: Choose Miami as the destination). \n\n" \
                "Subgoal 9: Go to view details to proceed to checkout.\n" \
                "Action 9-1: R9 = CLICK(R8, QUERY: Go to view details to proceed to checkout). \n\n" \
                "Subgoal 10: Apply package to help us find cheapest flight, hotel and room.\n" \
                "Action 10-1: R10 = CLICK(R9, QUERY: Apply package to help us find cheapest flight, hotel and room). \n\n" \
                "Subgoal 11: Proceed to checkout.\n" \
                "Action 11-1: R11 = CLICK(R10, QUERY: Proceed to checkout). \n\n" \
                "Example 2:\n" \
                "Task: Find a Ricky Kej track to listen and share which has been added in the last year and is between 2 to 10 minutes.\n\n" \
                "Natural language plan:\n" \
                "[searchbox]  Search -> TYPE: Ricky Kej; [link]  Search for “Ricky Kej” -> CLICK; [link]  Tracks -> CLICK; [link]  Added any time -> CLICK; [link]  Past year -> SELECT; [link]  Any length -> CLICK; [link]  2-10 min -> CLICK; [link]  To listen to -> CLICK; [link]  To share -> CLICK \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Type Ricky Kej to search his songs.\n" \
                "Action 1-1: R1 = TYPE(Env, QUERY: Type Ricky Kej to search his songs, TEXT: Ricky Kej). \n\n" \
                "Subgoal 2: Click on the option to search for Ricky Rej.\n" \
                "Action 2-1: R2 = CLICK(R1, QUERY: Click on the option to search for Ricky Rej). \n\n" \
                "Subgoal 3: Choose tracks as the search category.\n" \
                "Action 3-1: R3 = CLICK(R2, QUERY: Choose tracks as the search category). \n\n" \
                "Subgoal 4: Find the region to adjust the added time of our interested track.\n" \
                "Action 4-1: R4 = CLICK(R3, QUERY: Find the region to adjust the added time of our interested track). \n\n" \
                "Subgoal 5: Choose the last year as the added date.\n" \
                "Action 5-1: R5 = SELECT(R4, QUERY: Choose the last year as the added date, TEXT: Past year). \n\n" \
                "Subgoal 6: Find the region to adjust the track length of our interested track.\n" \
                "Action 6-1: R6 = CLICK(R5, QUERY: Find the region to adjust the track length of our interested track). \n\n" \
                "Subgoal 7: Choose 2 to 10 minutes as the track length.\n" \
                "Action 7-1: R7 = CLICK(R6, QUERY: Choose 2 to 10 minutes as the track length). \n\n" \
                "Subgoal 8: Listen to our searched track.\n" \
                "Action 8-1: R8 = CLICK(R7, QUERY: Listen to our searched track). \n\n" \
                "Subgoal 9: Share our searched track.\n" \
                "Action 9-1: R9 = CLICK(R8, QUERY: Share our searched track). \n\n" \
                "Now please help us generate a plan consisting of subgoals according to the following instruction and its natural language plan! \n\n"


instruction_icode = "Please convert programming language plans into a series of subgoals and their corresponding actions" \
                "that lead to the successful implementation with respect to the given instructions. " \
                "Please use 'R[number]' to represent the intermediate results for each subgoal, without generating" \
                "any exact values. Please also use functions to represent the corresponding actions. " \
                "For the actions, they must be one of the valid commands used in Linux operating systems.\n\n" \
                "Example 1:\n" \
                "Task: Counts lines in each *.php file in /testbed directory, sorted by number of lines, descending.\n\n" \
                "Programming language plan:\n" \
                "find /testbed -name '*.php' -type f | xargs wc -l | sort -nr \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Find all the files with filenames *.php in /testbed directory.\n" \
                "Action 1-1: R1 = find /testbed -name '*.php' -type f. \n\n" \
                "Subgoal 2: Sort the found files by number of lines, descending.\n" \
                "Action 2-1: R2 = find /testbed -name '*.php' -type f | xargs wc -l. \n" \
                "Action 2-2: R3 = find /testbed -name '*.php' -type f | xargs wc -l | sort -nr. \n\n" \
                "Example 2:\n" \
                "Task: Recursively removes all empty folders under /system/folder3/temp, printing info message on each operation, and suppressing error messages if folder is not empty.\n\n" \
                "Programming language plan:\n" \
                "find /system/folder3/temp -type d -empty -exec rmdir -vp --ignore-fail-on-non-empty {} + \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Find all the empty folders under /system/folder3/temp.\n" \
                "Action 1-1: R1 = find /system/folder3/temp -type d -empty. \n\n" \
                "Subgoal 2: Recursively remove all the founded folders and print info message on each operation.\n" \
                "Action 2-1: R2 = find /system/folder3/temp -type d -empty -exec rmdir -p. \n" \
                "Action 2-2: R3 = find /system/folder3/temp -type d -empty -exec rmdir -vp. \n\n" \
                "Subgoal 3: Suppress error messages if folder is not empty. \n" \
                "Action 3-1: R4 = find /system/folder3/temp -type d -empty -exec rmdir -vp --ignore-fail-on-non-empty {} +. \n\n" \
                "Example 3:\n" \
                "Task: Remove all but 5 last comma-separated fields from each line in '/system/folder1/data.csv'\n\n" \
                "Programming language plan:\n" \
                "cat /system/folder1/data.csv | rev | cut -d, -f-5 | rev \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Read data from the /system/folder1/data.csv.\n" \
                "Action 1-1: R1 = cat /system/folder1/data.csv. \n\n" \
                "Subgoal 2: Reverse all the data from each line.\n" \
                "Action 2-1: R2 = cat /system/folder1/data.csv | rev. \n\n" \
                "Subgoal 3: Only obtain the first 5 fields separated by comma for each reversed line. \n" \
                "Action 3-1: R3 = cat /system/folder1/data.csv | rev | cut -d, -f-5. \n\n" \
                "Subgoal 4: Reverse the 5 fields in Subgoal 4 back to the normal order. \n" \
                "Action 4-1: R4 = cat /system/folder1/data.csv | rev | cut -d, -f-5 | rev. \n\n" \
                "Example 4:\n" \
                "Task: Remove junk files modified more than 31 days ago recursively from \"/system\"\n\n" \
                "Programming language plan:\n" \
                "find /system -type f -mtime +31 -exec rm -f {} \\; \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Find all the files under /system.\n" \
                "Action 1-1: R1 = find /system -type f. \n\n" \
                "Subgoal 2: Find all the files under /system modified more than 31 days ago.\n" \
                "Action 2-1: R2 = find /system -type f -mtime +31. \n\n" \
                "Subgoal 3: Remove all the junk files among the files obtained in Subgoal 2. \n" \
                "Action 3-1: R3 = find /system -type f -mtime +31 -exec rm -f {} \\;. \n\n" \
                "Example 5:\n" \
                "Task: Calculate the md5sum of each \".txt\" file under \"/system\", sort the output, and calculate the md5sum of that\n\n" \
                "Programming language plan:\n" \
                "find /system -type f -name '*.txt' -exec md5sum {} + | awk '{print $1}' | sort | md5sum \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Find all the \".txt\" files under \"/system\" and calculate their md5sum.\n" \
                "Action 1-1: R1 = find /system -type f -name '*.txt'. \n" \
                "Action 1-2: R2 = find /system -type f -name '*.txt' -exec md5sum {} + . \n\n" \
                "Subgoal 2: Sort the files based on their md5sum.\n" \
                "Action 2-1: R3 = find /system -type f -name '*.txt' -exec md5sum {} + | awk '{print $1}'. \n\n" \
                "Subgoal 3: Calculate the md5sum.\n" \
                "Action 3-1: R4 = find /system -type f -name '*.txt' -exec md5sum {} + | awk '{print $1}' | sort | md5sum. \n\n" \
                "Now please help us generate a plan consisting of subgoals according to the following instruction and its programming language plan! \n\n"


instruction_icode_sql = "Please convert natural language plans into a series of subgoals and their corresponding actions" \
                "that lead to the successful implementation with respect to the given instructions. " \
                "Please use 'R[number]' to represent the intermediate results for each subgoal, without generating" \
                "any exact values. Please also use functions to represent the corresponding actions. " \
                "For the actions, they must be one of the valid operations used in SQL.\n\n" \
                "Example 1:\n" \
                "Task: Find the name of airports which do not have any flight in and out.\n\n" \
                "Natural language plan:\n" \
                "We have 3 SQL tables: airlines, airports, flights. Airlines table has columns: uid (type: INT), Airline (type: INT), Abbreviation (type: TEXT), Country (type: TEXT). Airports table has columns: City (type: TEXT), AirportCode (type: VARCHAR(255)), AirportName (type: TEXT), Country (type: TEXT), CountryAbbrev (type: TEXT). Flights table has columns: Airline (type: INT), FlightNo (type: INT), SourceAirport (type: VARCHAR(255)), DestAirport (type: VARCHAR(255)). The gold SQL query for this task is SELECT AirportName FROM Airports WHERE AirportCode NOT IN (SELECT SourceAirport FROM Flights UNION SELECT DestAirport FROM Flights). \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Find all the related tables.\n" \
                "Action 1-1: R1 = SHOW TABLES = airlines, airports, flights. \n\n" \
                "Subgoal 2: Describe the columns for all the related tables.\n" \
                "Action 2-1: R2 = DESCRIBE airlines = uid (type: INT), Airline (type: INT), Abbreviation (type: TEXT), Country (type: TEXT). \n" \
                "Action 2-2: R3 = DESCRIBE airports = City (type: TEXT), AirportCode (type: VARCHAR(255)), AirportName (type: TEXT), Country (type: TEXT), CountryAbbrev (type: TEXT). \n" \
                "Action 2-3: R4 = DESCRIBE flights = Airline (type: INT), FlightNo (type: INT), SourceAirport (type: VARCHAR(255)), DestAirport (type: VARCHAR(255)). \n\n" \
                "Subgoal 3: Write the selection objects: the airports in airports table.\n" \
                "Action 3-1: R5 = AirportName FROM airports. \n\n" \
                "Subgoal 4: Write the selection condition: the queried airports should be neither the source airports nor destination airports for all the flights.\n" \
                "Action 4-1: R6 = AirportCode NOT IN (SELECT SourceAirport FROM Flights UNION SELECT DestAirport FROM Flights). \n\n" \
                "Subgoal 5: Finish the SQL query to select the name of airports satisfying the previous condition.\n" \
                "Action 5-1: R7 = SELECT AirportName FROM airports WHERE AirportCode NOT IN (SELECT SourceAirport FROM Flights UNION SELECT DestAirport FROM Flights). \n\n" \
                "Example 2:\n" \
                "Task: What is the average GNP and total population in all nations whose government is US territory?\n\n" \
                "Natural language plan:\n" \
                "We have 1 SQL tables: country. Country table has columns: Code (type: CHAR(3)), Name (type: CHAR(52)), Continent (type: TEXT), Region (type: CHAR(26)), Population (type: INT), GNP (type: FLOAT(10,2)), GovernmentForm (type: CHAR(45)). The gold SQL query for this task is SELECT avg(GNP) , sum(Population) FROM country WHERE GovernmentForm = \"US Territory\". \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Find all the related tables.\n" \
                "Action 1-1: R1 = SHOW TABLES = country. \n\n" \
                "Subgoal 2: Describe the columns for all the related tables.\n" \
                "Action 2-1: R2 = DESCRIBE country = Code (type: CHAR(3)), Name (type: CHAR(52)), Continent (type: TEXT), Region (type: CHAR(26)), Population (type: INT), GNP (type: FLOAT(10,2)), GovernmentForm (type: CHAR(45)). \n\n" \
                "Subgoal 3: Write the selection objects: the average GNP and total population in country table.\n" \
                "Action 3-1: R3 = avg(GNP) , sum(Population) FROM country. \n\n" \
                "Subgoal 4: Write the selection condition: the countries' government should be US territory.\n" \
                "Action 4-1: R4 = GovernmentForm = \"US Territory\". \n\n" \
                "Subgoal 5: Finish the SQL query to calculate the average GNP and total population in all nations satisfying the previous condition.\n" \
                "Action 5-1: R5 = SELECT avg(GNP) , sum(Population) FROM country WHERE GovernmentForm = \"US Territory\". \n\n" \
                "Example 3:\n" \
                "Task: How many pets are owned by students that have an age greater than 20?\n\n" \
                "Natural language plan:\n" \
                "We have 3 SQL tables: student, has_pet, pets. Student table has columns: stuid (type: INT), age (type: INT). Has_pet table has columns: stuid (type: INT), petid (type: INT). Pets table has columns: petid (type: INT), pet_age (type: INT). The gold SQL query for this task is SELECT COUNT(*) FROM has_pet INNER JOIN student ON has_pet.stuid = student.stuid WHERE student.age > 20. \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Find all the related tables.\n" \
                "Action 1-1: R1 = SHOW TABLES = student, has_pet, pets. \n\n" \
                "Subgoal 2: Describe the columns for all the related tables.\n" \
                "Action 2-1: R2 = DESCRIBE student = stuid (type: INT), age (type: INT). \n" \
                "Action 2-1: R3 = DESCRIBE has_pet = stuid (type: INT), petid (type: INT). \n" \
                "Action 2-1: R4 = DESCRIBE pets = petid (type: INT), pet_age (type: INT). \n\n" \
                "Subgoal 3: Write the selection objects: the number of pets owned by students.\n" \
                "Action 3-1: R5 = COUNT(*) FROM has_pet INNER JOIN student ON has_pet.stuid = student.stuid. \n\n" \
                "Subgoal 4: Write the selection condition: students should have an age greater than 20.\n" \
                "Action 4-1: R6 = student.age > 20. \n\n" \
                "Subgoal 5: Finish the SQL query to calculate the number of pets owned by students satisfying the previous condition.\n" \
                "Action 5-1: R7 = SELECT COUNT(*) FROM has_pet INNER JOIN student ON has_pet.stuid = student.stuid WHERE student.age > 20. \n\n" \
                "Example 4:\n" \
                "Task: Sort all the shops by number products in descending order, and return the name, location and district of each shop.\n\n" \
                "Natural language plan:\n" \
                "We have 2 SQL tables: employee, shop. Employee table has columns: Name (type: TEXT), Age (type: INT), City (type: TEXT). Shop table has columns: Name (type: TEXT), Location (type: TEXT), District (type: TEXT), Number_products (type: INT). The gold SQL query for this task is SELECT Name, Location, District FROM shop ORDER BY Number_products DESC. \n\n" \
                "Subgoal-based plan:\n" \
                "Subgoal 1: Find all the related tables.\n" \
                "Action 1-1: R1 = SHOW TABLES = employee, shop. \n\n" \
                "Subgoal 2: Describe the columns for all the related tables.\n" \
                "Action 2-1: R2 = DESCRIBE employee = Name (type: TEXT), Age (type: INT), City (type: TEXT). \n" \
                "Action 2-2: R3 = DESCRIBE shop = Name (type: TEXT), Location (type: TEXT), District (type: TEXT), Number_products (type: INT). \n\n" \
                "Subgoal 3: Write the selection objects: the name, location and district in shop table.\n" \
                "Action 3-1: R4 = Name, Location, District FROM shop. \n\n" \
                "Subgoal 4: Write the selection condition: sort all the shops by number products in descending order.\n" \
                "Action 4-1: R5 = ORDER BY Number_products DESC. \n\n" \
                "Subgoal 5: Finish the SQL query to return the name, location and district of the shops after sorting.\n" \
                "Action 5-1: R6 = SELECT Name, Location, District FROM shop ORDER BY Number_products DESC. \n\n" \
                "Now please help us generate a plan consisting of subgoals according to the following instruction and its natural language plan! \n\n"