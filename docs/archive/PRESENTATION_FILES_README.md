# Presentation Files Guide

## Which File Should I Use?

You have **3 main presentation files**. Here's what each one is for:

---

## 📊 **PRESENTATION_SLIDES_SHORT.Rmd** ⭐ **USE THIS ONE!**

**File:** `docs/PRESENTATION_SLIDES_SHORT.Rmd` (27KB)

**What it is:**
- R Markdown file with **code chunks** and **detailed explanations**
- 14 slides, conversational tone
- Includes code with comments explaining WHAT, WHY, WHERE
- Has visual highlights for important code sections

**When to use:**
- ✅ **For your presentation** - Render this to create slides
- ✅ **When you need code examples** - Shows actual code with explanations
- ✅ **For detailed explanations** - Every code block has comments

**How to use:**
```r
# In R or RStudio
rmarkdown::render("docs/PRESENTATION_SLIDES_SHORT.Rmd")
```

**Output:** Creates HTML or PDF slides with all code chunks visible

---

## 📝 **PRESENTATION_SLIDES_SHORT.md**

**File:** `docs/PRESENTATION_SLIDES_SHORT.md` (7.8KB)

**What it is:**
- Simple Markdown version (no code chunks)
- 14 slides, same content as Rmd but without R/Python code
- Easy to read in any text editor
- Quick reference version

**When to use:**
- ✅ **Quick reference** - Read without rendering
- ✅ **Copy/paste content** - Easy to extract text
- ✅ **Backup version** - If Rmd doesn't work

**Note:** This is a simplified version. Use the `.Rmd` file for the full presentation.

---

## 📚 **PRESENTATION_SLIDES.md**

**File:** `docs/PRESENTATION_SLIDES.md` (21KB)

**What it is:**
- Full detailed version with 20 slides
- More comprehensive coverage
- Includes all details and explanations

**When to use:**
- ✅ **Reference material** - More detailed explanations
- ✅ **If you need more slides** - Has additional content
- ⚠️ **Not for presentation** - Too long (20 slides vs 14)

**Note:** This is the comprehensive version. Use `PRESENTATION_SLIDES_SHORT.Rmd` for your actual presentation.

---

## 🗑️ **Files to Ignore:**

- `PRESENTATION_SLIDES_SHORT.html` - **Auto-generated** (don't commit, will be recreated)
  - This is created when you render the `.Rmd` file
  - Already in `.gitignore`

---

## 📋 **Quick Decision Guide:**

| Need | Use This File |
|------|---------------|
| **Give presentation** | `PRESENTATION_SLIDES_SHORT.Rmd` ⭐ |
| **Quick read** | `PRESENTATION_SLIDES_SHORT.md` |
| **More details** | `PRESENTATION_SLIDES.md` |
| **Code examples** | `PRESENTATION_SLIDES_SHORT.Rmd` ⭐ |

---

## ✅ **Recommended Setup:**

1. **For P3 Submission:** Use `PRESENTATION_SLIDES_SHORT.Rmd`
2. **Render it** to create your presentation slides
3. **Keep the other files** as backup/reference

---

## 🎯 **Summary:**

- **Main file:** `PRESENTATION_SLIDES_SHORT.Rmd` (use this!)
- **Quick reference:** `PRESENTATION_SLIDES_SHORT.md`
- **Detailed version:** `PRESENTATION_SLIDES.md` (reference only)
- **Ignore:** `*.html` files (auto-generated)

**Bottom line:** Use `PRESENTATION_SLIDES_SHORT.Rmd` for your presentation! 🎤

