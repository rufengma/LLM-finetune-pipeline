# Evaluation Results

_Generated: 2026-09-22 16:30 UTC · base model: `gpt2` · adapter: `outputs/smoke-test`_

## Held-out perplexity (response tokens only)

| Model | Mean NLL | Perplexity |
|-------|----------|------------|
| Base | 3.6752 | 39.46 |
| LoRA-tuned | 3.4998 | 33.11 |

## Sample generations (greedy decoding)
### Sample 1

**Instruction:** Write a short apology email for missing a deadline.

| | |
|---|---|
| **Base** | Dear Mr. Smith,<br><br>I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. |
| **LoRA-tuned** | Dear Mr. Smith,<br><br>I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. I am sorry for the inconvenience. |
| **Reference** | Subject: Update on the report timeline<br><br>Hi Priya,<br><br>I owe you an apology — the report won't be ready by today as promised. I ran into an unexpected data issue yesterday. I've resolved it and will have the final version to you by tomorrow at noon. I understand this affects your planning, and I'll keep you posted if anything changes.<br><br>Best,<br>Lee |
### Sample 2

**Instruction:** What is gradient descent in one paragraph?

| | |
|---|---|
| **Base** | The following is a response to a question about gradient descent in one paragraph.<br><br>### Instruction:<br><br>What is gradient descent in one paragraph?<br><br>### Response:<br><br>The following is a response to a question about gradient descent in one paragraph.<br><br>### Instruction:<br><br>What is gradient descent |
| **LoRA-tuned** | The gradient descent algorithm is a simple, but powerful, way to learn how to apply gradient descent to a problem. It is a simple, but powerful, way to learn how to apply gradient descent to a problem. It is a simple, but powerful, way to learn how to apply gradient descent to a problem. |
| **Reference** | Gradient descent is an optimization algorithm that minimizes a function by repeatedly stepping in the direction of steepest decrease. For a model, it computes how each parameter contributes to the error (the gradient) and nudges parameters against that gradient by a small learning rate, iterating until the error stops improving. |
### Sample 3

**Instruction:** What are the key sections of a good resume for a software engineer?

| | |
|---|---|
| **Base** | The key sections of a good resume are:<br><br>What are the key sections of a good resume for a software engineer?<br><br>### Response:<br><br>The key sections of a good resume are:<br><br>What are the key sections of a good resume for a software engineer?<br><br>### Response: |
| **LoRA-tuned** | The key sections of a good resume are:<br><br>What are the key areas of a good resume for a software engineer?<br><br>### Response:<br><br>The key areas of a good resume are:<br><br>What are the key areas of a good resume for a software engineer?<br><br>### Response: |
| **Reference** | 1. Contact info and a one-line summary of your focus.<br>2. Skills: languages, frameworks, and tools, grouped by category.<br>3. Experience: 2-4 bullets per role with quantified impact (e.g., 'cut API latency 40%').<br>4. Projects: 1-2 standout projects with links.<br>5. Education. Keep it to one page and tailor keywords to the job posting. |
### Sample 4

**Instruction:** List three differences between TCP and UDP.

| | |
|---|---|
| **Base** | The TCP protocol is a TCP protocol. It is a protocol that is used to communicate with other TCP protocols. It is used to communicate with other TCP protocols. It is used to communicate with other TCP protocols. It is used to communicate with other TCP protocols. It is used to communicate with other TCP protocols. It |
| **LoRA-tuned** | The TCP protocol is a TCP protocol. It is a protocol that is used to communicate with other TCP protocols. It is used to communicate with other TCP protocols. It is used to communicate with other TCP protocols. It is used to communicate with other TCP protocols. It is used to communicate with other TCP protocols. It |
| **Reference** | 1. Reliability: TCP guarantees delivery with acknowledgments and retransmission; UDP sends packets without guarantees.<br>2. Ordering: TCP delivers data in order; UDP packets may arrive out of order or be lost.<br>3. Speed/overhead: UDP is faster with less overhead, which suits live video and gaming, while TCP suits file transfers and web browsing where correctness matters. |
### Sample 5

**Instruction:** Write a checklist for launching a personal blog.

| | |
|---|---|
| **Base** | Write a response to your question.<br><br>### Conclusion:<br><br>Write a response to your question.<br><br>###<br><br>This is a great tool for beginners. It's easy to use, and it's easy to learn. It's also a great way to learn about the world of blogging. |
| **LoRA-tuned** | Write a response to your question.<br><br>### Conclusion:<br><br>Write a response to your question.<br><br>###<br><br>This post was written by a member of the community. |
| **Reference** | 1. Pick a topic and a simple platform (static site or hosted).<br>2. Choose a clean, readable theme.<br>3. Write 3-5 posts before announcing, so visitors have something to read.<br>4. Set up an RSS feed and basic analytics.<br>5. Share the first post and commit to a realistic schedule (e.g., biweekly). |
### Sample 6

**Instruction:** What is the main idea of this paragraph?<br>Input: Urban trees do more than beautify streets. Studies show they can lower summer air temperatures by several degrees, absorb stormwater runoff, filter air pollutants, and even reduce stress levels in nearby residents. Cities that invest in canopy cover…

| | |
|---|---|
| **Base** | Urban trees are a critical part of the urban fabric. They help to protect the environment and provide a safe, clean, and efficient way to live. They are also a critical part of the fabric of our cities.<br><br>### Conclusion:<br><br>Urban trees are a critical part of the fabric of our cities. |
| **LoRA-tuned** | The Urban Tree Project is committed to providing a safe, sustainable, and environmentally friendly way to grow trees. We are committed to providing a safe, sustainable, and environmentally friendly way to grow trees. We are committed to providing a safe, sustainable, and environmentally friendly way to grow trees. We are committed to providing a |
| **Reference** | Urban trees provide significant environmental and health benefits, from cooling cities and managing stormwater to improving air quality and residents' well-being. |
### Sample 7

**Instruction:** Summarize the following paragraph in one sentence.<br>Input: Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen. Chlorophyll, the pigment that gives plants their green color, captures light energy, which powers a series of chemical…

| | |
|---|---|
| **Base** | The following paragraph is a response to the following question:<br><br>What is the most important thing you can do to help your plants grow?<br><br>The answer is:<br><br>The most important thing you can do to help your plants grow is to grow plants that are healthy.<br><br>### Conclusion: |
| **LoRA-tuned** | The photosynthetic process is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. Chlorophyll is a pigment that is used to make photosynthesis. It is a pigment that is used to make photosynthesis. It is a pigment that is used to make photosynthesis. |
| **Reference** | Plants use chlorophyll to convert sunlight, water, and carbon dioxide into glucose and oxygen through photosynthesis. |
### Sample 8

**Instruction:** What is the difference between authentication and authorization?

| | |
|---|---|
| **Base** | Authentication is the process of authenticating a user to a website. Authorization is the process of authenticating a user to a website.<br><br>### Response:<br><br>Authorization is the process of authenticating a user to a website. Authorization is the process of authenticating a user to a website. |
| **LoRA-tuned** | Authentication is the process of authenticating a user to a website. Authorization is the process of authenticating a user to a website. Authorization is the process of authenticating a user to a website.<br><br>### Response:<br><br>Authentication is the process of authenticating a user to a website. Authorization is |
| **Reference** | Authentication answers 'who are you?' — verifying identity via a password, key, or biometrics. Authorization answers 'what are you allowed to do?' — checking permissions after identity is established. Logging in is authentication; being allowed to view the admin dashboard is authorization. |
