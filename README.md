I need your help to clean up the codebase.

1. I need research first, fail early code. No verbose try/catch patterns plain asserts with a short message is enough
2. Migrate usefull and necessary utils from ai4bmr_core and delete if after.
2. I need well structured, concise, easy to read code. The Zen of Python encapsulates all this precisely (see below)

With this in mind, take a holistic view at the codebase. Restructure, simplify, refactor the codebase to reflect the Zen of Python.

Zen of Python:
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

Principles for the code:I need research first
1. Research first.
2. Fail early and never silently
3. Be concise
4. Use informative but not verbose function names
4. Use asserts with a short precise error message to enforce assumptions, don't write verbose try/catch
3. Use patterns that align with intent, this includes `match` for switching between cases with raising assertions