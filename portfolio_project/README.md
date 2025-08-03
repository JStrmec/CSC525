# CSC525 Portfolio Project Retrival Chatbot

To download the model wieghts from the public bucket:
```
gsutil -m cp -r \
  "gs://csc525-portfolio-project/empathetic-finetuned-chatbot" \
  .
```

To run the program use `make run`. Or `python mental_health_retrieval_chatbot/ui.py`

The UI will then be running on http://127.0.0.1:7860. There will be a text box where you can start your conversation.

A few easy prompts to start with include:
 - I feel anxious. I can’t sleep and struggle to rest properly.
 - Is there any activity I can do to get me out of my funk?
 - I'm feeling really down lately.
 - Nothing seems to help me anymore.
 - Is there even a point to trying again?
 - I am struggling with school, what is the best way to get help?