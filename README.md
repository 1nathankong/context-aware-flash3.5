# context-aware-flash3.5

Idea was to address token useage of 3.5 Flash model. 

1. Plan was to create a script to stress test the 3.5 Flash model on prompts that require alot of token generation and check if the built in libaries used caching accordingly to the documentation.
2. Then connect it to a dashboard that the user could see internally, locally secure so all data is exclusive to the user.
3. Present and pitch to judges, goal of this progress was to adress cloud llms and how to help companies who deploy google products manage token useage from a non technical standpoint. 


Some observations.

Even though I did not make progress I have some feedback.

When building a internal dashboard for personal user, I got rate limited very early into developing the project. Google AI studio uses alot of tokens to generation the front and back end even though no actual coding is required.

I observed that prompt caching worked as intendended and as prompts got larger more tokens and memory got saved overtime, which means that token and memory caching works as intended as the documentation states online. 


I am grateful for trying new google tools, I just wish I had more tokens to keep building and finding more use cases to improve for the tools. 


Local Gemma model cache testing results:

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/854bb5b9-8412-4127-95f0-f8911a7fe9cf" />

