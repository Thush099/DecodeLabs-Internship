import random

print("System Initialized. Type 'help' for a list of topics, or 'exit' to power down.\n")

while True:

    raw_input = input('You: ')
    clean_input = raw_input.lower().strip()

    # GREETINGS
    if clean_input in ['hello', 'hi', 'hey', 'start', 'greetings']:
        greetings = [
            "Bot: Hello! Welcome to the DecodeLabs team. Ready to build systems?",
            "Bot: Hi there! Deterministic System standing by.",
            "Bot: Greetings, Engineer! Awaiting your input."
        ]
        print(random.choice(greetings))

    # PROJECT 1 GOALS
    elif 'project' in clean_input or 'goal' in clean_input:
        project_responses = [
            "Bot: Project 1 is your foundation phase. The goal is to create a simple rule-based chatbot that responds to predefined user inputs.",
            "Bot: This track isn't about deep learning yet! It's about mastering Control Flow and Logic by teaching a machine through explicit if-else instructions."
        ]
        print(random.choice(project_responses))

    #  SYSTEM 1 VS SYSTEM 2
    elif 'system 1' in clean_input or 'system 2' in clean_input or 'artist' in clean_input:
        mind_responses = [
            "Bot: There are two minds of AI. System 1 is 'The Artist' (Probabilistic), and System 2 is 'The Engineer' (Deterministic). I am a System 2 logic engine.",
            "Bot: Before you can manage the chaos of a probability engine (System 1), you must master the precision of a logic engine (System 2)!"
        ]
        print(random.choice(mind_responses))

    # THE WHITE BOX & HALLUCINATIONS
    elif 'white box' in clean_input or 'hallucination' in clean_input or 'safe' in clean_input:
        whitebox_responses = [
            "Bot: As a 'White Box' program, my logic is fully traceable (Input -> Logic -> Output). There is zero mystery.",
            "Bot: I am 100% hard-coded, which means I have zero hallucination risk! This safety is essential for fields like Finance and Healthcare."
        ]
        print(random.choice(whitebox_responses))

    #  INTENT: AI GUARDRAILS
    elif 'guardrail' in clean_input or 'nemo' in clean_input or 'llama' in clean_input:
        print("Bot: Rule-based AI acts as a deterministic filter (guardrail) for probabilistic outputs. Frameworks like NVIDIA NeMo and Llama Guard reside in this control layer!")

    #  THE IPO MODEL
    elif 'ipo' in clean_input or 'blueprint' in clean_input:
        print("Bot: The IPO blueprint stands for Input (Raw Feed), Process (Intent Matching & Logic Skeleton), and Output (Response Generation). You are currently testing my Output phase.")

    # HELP MENU
    elif 'help' in clean_input or 'options' in clean_input:
        print("\nBot: Here is what I am trained to discuss based on DecodeLabs Module 01:")
        print("     - Greetings (hi, hello)")
        print("     - Project Goals (what is the goal of project 1?)")
        print("     - AI Minds (what is system 1 and system 2?)")
        print("     - The White Box (what is a white box? / hallucinations)")
        print("     - AI Guardrails (what are guardrails?)")
        print("     - The IPO Blueprint (what is IPO?)")
        print("     - 'exit' to quit the program\n")

    #  EXIT COMMANDS
    elif clean_input in ['exit', 'quit', 'bye', 'goodbye']:
        goodbyes = [
            "Bot: Powering down. Goodbye!",
            "Bot: Shutting down deterministic systems. See you next time!",
            "Bot: Terminating continuous loop. Have a great day!"
        ]
        print(random.choice(goodbyes))
        break

    # FALLBACK
    else:
        fallbacks = [
            "Bot: System 2 engaged. I am a deterministic bot and only understand explicit, specific rules right now.",
            "Bot: I didn't quite catch that. Type 'help' to see my supported topics.",
            "Bot: Error: Intent not found. As a White Box AI, my explicit if-else instructions don't cover that yet!"
        ]
        print(random.choice(fallbacks))