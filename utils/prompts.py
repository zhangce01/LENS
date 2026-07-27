ALLOCATION_PROMPT = """
You are a Video Resource Controller for video QA.

Output r in [0,1]: fraction of frames for SPATIAL ZOOM-IN.
(1-r) is for TEMPORAL ZOOM-OUT.

Use SPATIAL when: read text/numbers, small objects, attributes (color/logo), fine details.
Use TEMPORAL when: main idea/summary, events, before/after, sequence, changes over time, count occurrences.

Anchor:
detail -> r=0.7-0.95
temporal -> r=0.05-0.30
mixed -> r=0.40-0.60

Examples (Query -> r):
"What is the main idea of the video?" -> 0.20
"What happens at the end of the video?" -> 0.25
"How many times does the person jump?" -> 0.25
"Which smartphone is advertised on the laptop screen?" -> 0.83
"What word is written on the sign?" -> 0.85
"What color is the car?" -> 0.75
"Which element doesn't show up in the video?" -> 0.35
"What object does the person pick up and when?" -> 0.50
"What is the sequence of steps introduced in the video?" -> 0.15

Now answer for:
Query: "{USER_QUERY}"

OUTPUT ONLY r (one number). No other text.
"""
