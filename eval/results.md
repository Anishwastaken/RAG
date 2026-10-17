# Retrieval Evaluation Results

42 questions over 1870 chunks. A hit means a retrieved chunk contains the answer's evidence phrase.

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| vector | 52.4% | 73.8% | 78.6% | 0.659 |
| bm25 | 57.1% | 81.0% | 81.0% | 0.677 |
| hybrid | 64.3% | 83.3% | 92.9% | 0.754 |
| hybrid+mq | 54.8% | 78.6% | 85.7% | 0.677 |

## Rank of first correct chunk per question (– = not in top 10)

| ID | Question | vector | bm25 | hybrid | hybrid+mq |
|---|---|---|---|---|---|
| goog-01 | At what price per share did Google sell stock in its IPO? | 1 | 2 | 1 | 1 |
| goog-02 | How much of their working time are Google engineers encouraged to spend on side projects? | 1 | 1 | 1 | 1 |
| goog-03 | Which company did Google replace as Yahoo's default search provider? | 1 | 2 | 1 | 1 |
| goog-04 | What is the address of the office complex Google leased from Silicon Graphics? | 6 | 1 | 2 | 3 |
| goog-05 | By what year does Google want its data centers and offices to run only on carbon-free energy? | 2 | 1 | 1 | 1 |
| goog-06 | How large was the fine Europe's top court imposed on Google over shopping comparison? | 7 | 1 | 5 | 7 |
| goog-07 | What share of Google's workforce were women according to the 2020 diversity report? | 1 | 1 | 1 | 1 |
| goog-08 | How much did Google reportedly pay to bring Ubisoft games to Stadia? | 2 | 1 | 1 | 1 |
| msft-01 | What product was Microsoft originally founded to develop and sell? | 1 | 1 | 1 | 1 |
| msft-02 | Who succeeded Bill Gates as Microsoft's chief executive? | 8 | 6 | 5 | 4 |
| msft-03 | Which custom chip did Microsoft design for running large language models? | 1 | 1 | 1 | 1 |
| msft-04 | How many HoloLens headsets did Microsoft agree to supply to the US military? | 1 | 1 | 1 | 1 |
| msft-05 | How many jobs did Satya Nadella say Microsoft would cut in January 2023? | 1 | 1 | 1 | 1 |
| msft-06 | What stake in the London Stock Exchange Group did Microsoft acquire? | – | 3 | – | 10 |
| msft-07 | How much did Microsoft spend on carbon credits from Heirloom Carbon? | 1 | 1 | 1 | 1 |
| msft-08 | What was the name of Bill Gates's 1995 internal memo about the internet? | 1 | 1 | 1 | 1 |
| msft-09 | Why did the European Union fine Microsoft again in February 2008? | 1 | 2 | 1 | 2 |
| nvda-01 | Where does the name Nvidia come from? | 7 | 3 | 2 | 6 |
| nvda-02 | Why couldn't the founders name the company NVision? | 3 | 1 | 1 | 6 |
| nvda-03 | What was Nvidia's first graphics accelerator called? | 1 | 1 | 1 | 1 |
| nvda-04 | Which physics engine developer did Nvidia buy in 2008? | 1 | 1 | 1 | 1 |
| nvda-05 | On what date did Nvidia overtake Microsoft and Apple as the most valuable company? | 2 | – | 2 | 5 |
| nvda-06 | What caused Nvidia's record single-day loss in market value in January 2025? | 4 | 7 | 3 | 3 |
| nvda-07 | In which restaurant was Nvidia founded? | 4 | – | 4 | 3 |
| nvda-08 | How many members did the Inception startup program have in 2021? | 1 | 1 | 1 | 1 |
| spcx-01 | How many launches failed before Falcon 1 finally reached orbit? | – | 3 | – | – |
| spcx-02 | What was the total value of SpaceX's COTS development contract? | – | – | – | – |
| spcx-03 | What did the Starlink Mini antenna cost at its early access release? | 2 | 1 | 1 | 1 |
| spcx-04 | According to NASA, how much did SpaceX spend developing the Falcon 9? | 2 | 3 | 2 | 2 |
| spcx-05 | What is the maximum payload Falcon 9 can deliver to orbit? | 1 | 3 | 1 | 1 |
| spcx-06 | How long is SpaceX's lease on Launch Complex 39A? | 2 | 1 | 1 | 4 |
| spcx-07 | Who paid for the Inspiration4 private spaceflight? | 1 | 3 | 1 | 1 |
| spcx-08 | What went wrong on Starship's seventh test flight? | 2 | – | 5 | 2 |
| spcx-09 | What is the name of the drone ship SpaceX unveiled in July 2021? | 1 | 1 | 1 | 1 |
| tsla-01 | Who incorporated Tesla Motors in 2003? | 1 | 8 | 1 | 1 |
| tsla-02 | How much bitcoin did Tesla buy in early 2021? | 2 | 1 | 2 | 2 |
| tsla-03 | For how long did Tesla accept bitcoin as payment before stopping? | 1 | 1 | 1 | 2 |
| tsla-04 | Which factory did Tesla take over through the SolarCity acquisition? | 7 | 1 | 2 | 3 |
| tsla-05 | What civil penalties did Tesla and Musk pay after the take-private tweet? | 1 | 1 | 1 | 1 |
| tsla-06 | What was Tesla's share of the battery-electric car market in 2024? | 1 | 2 | 1 | 1 |
| tsla-07 | What is the maximum charging power of a Tesla Supercharger? | 1 | – | 2 | 1 |
| tsla-08 | Which TV programme did Tesla lend two Roadsters to for testing in 2008? | 7 | 1 | 1 | 3 |
