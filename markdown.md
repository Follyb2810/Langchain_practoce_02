# 🔑 Basic Syntax

## 1. Headings

Use `#` symbols (1–6 levels).

```markdown
# Heading 1
## Heading 2
### Heading 3
#### Heading 4
```

👉 Rendered:

# Heading 1

## Heading 2

### Heading 3

#### Heading 4

---

## 2. Bold & Italic

* **Bold:** `**text**`
* *Italic:* `*text*`
* ***Bold + Italic:*** `***text***`

👉 Example:

```markdown
This is **bold**, this is *italic*, and this is ***bold + italic***.
```

---

## 3. Lists

### Bullet list

```markdown
- Item 1
- Item 2
  - Subitem
  - Subitem
```

👉 Rendered:

* Item 1
* Item 2

  * Subitem
  * Subitem

### Numbered list

```markdown
1. First
2. Second
3. Third
```

👉 Rendered:

1. First
2. Second
3. Third

---

## 4. Links & Images

* **Link:** `[title](https://example.com)`
* **Image:** `![alt text](https://example.com/image.png)`

👉 Example:

```markdown
Visit [Google](https://google.com)

![Cat image](https://placekitten.com/200/300)
```

---

## 5. Blockquotes

Use `>` for quotes.

```markdown
> This is a blockquote.
> It can span multiple lines.
```

👉 Rendered:

> This is a blockquote.
> It can span multiple lines.

---

## 6. Code

### Inline code

```markdown
Use `print("hello")` in Python.
```

👉 Rendered: Use `print("hello")` in Python.

### Code block

Use triple backticks (\`\`\`).

````markdown
```python
def hello():
    print("Hello World")
````

````

👉 Rendered:

```python
def hello():
    print("Hello World")
````

---

## 7. Tables

```markdown
| Name   | Age | Job       |
|--------|-----|-----------|
| Alice  | 24  | Engineer  |
| Bob    | 30  | Designer  |
```

👉 Rendered:

| Name  | Age | Job      |
| ----- | --- | -------- |
| Alice | 24  | Engineer |
| Bob   | 30  | Designer |

---

## 8. Horizontal Line

Use `---` or `***`.

```markdown
---
```

👉 Rendered:

---

---

# 🚀 Example Document in Markdown

````markdown
# My Project

## Overview
This is **my project**. It helps with:
- Simplicity
- Flexibility
- Speed

## Installation
```bash
npm install myproject
````

## Usage

```python
from myproject import run
run()
```


# 📘 Markdown Cheat Sheet

## 🏷️ Headings

```markdown
# H1
## H2
### H3
#### H4
##### H5
###### H6
```

👉 Example:

# H1

## H2

### H3

---

## ✨ Text Styling

```markdown
**Bold**  
*Italic*  
***Bold + Italic***  
~~Strikethrough~~  
```

👉 Example:
**Bold**
*Italic*
***Bold + Italic***
~~Strikethrough~~

---

## 📋 Lists

**Unordered:**

```markdown
- Item 1
- Item 2
  - Subitem
```

**Ordered:**

```markdown
1. First
2. Second
3. Third
```

👉 Example:

* Item 1
* Item 2

  * Subitem

1. First
2. Second
3. Third

---

## 🔗 Links & Images

```markdown
[Google](https://google.com)  
![Alt Text](https://placekitten.com/200/300)
```

👉 Example:
[Google](https://google.com)
![Alt Text](https://placekitten.com/200/150)

---

## 💬 Quotes

```markdown
> This is a quote.
```

👉 Example:

> This is a quote.

---

## 💻 Code

**Inline:**

```markdown
Use `print("hello")` in Python.
```

**Block:**

````markdown
```python
def hello():
    print("Hello World")
````

````

👉 Example:  
`print("hello")`

```python
def hello():
    print("Hello World")
````

---

## 📊 Tables

```markdown
| Name   | Age | Job       |
|--------|-----|-----------|
| Alice  | 24  | Engineer  |
| Bob    | 30  | Designer  |
```

👉 Example:

| Name  | Age | Job      |
| ----- | --- | -------- |
| Alice | 24  | Engineer |
| Bob   | 30  | Designer |

---

## ➖ Line Breaks

```markdown
---
```
