#include <cosmico/ui/SciencePanel.h>
#include <cosmico/simulation/Simulation.h>
#include <imgui.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>

namespace cosmico {

// =====================================================================
//  Init
// =====================================================================

void SciencePanel::init(ImFont* body, ImFont* heading, ImFont* formula,
                        ImFont* bodyZoom[], ImFont* headingZoom[], ImFont* formulaZoom[],
                        const std::string& papersDir) {
    m_fontBody    = body;
    m_fontHeading = heading;
    m_fontFormula = formula;
    for (int i = 0; i < ZOOM_LEVELS; i++) {
        m_fontBodyZoom[i]    = bodyZoom[i];
        m_fontHeadingZoom[i] = headingZoom[i];
        m_fontFormulaZoom[i] = formulaZoom[i];
    }
    m_papersDir   = papersDir;
}

void SciencePanel::setPapersDir(const std::string& dir) {
    if (dir != m_papersDir) {
        m_papersDir = dir;
        // Force reload of all documents
        m_nbody.blocks.clear();
        m_pm.blocks.clear();
        m_inflation.blocks.clear();
        m_cdt2d.blocks.clear();
        m_cdt3d.blocks.clear();
    }
}

// =====================================================================
//  LaTeX → Unicode conversion
// =====================================================================

static const std::unordered_map<std::string, std::string>& latexSymbols() {
    static const std::unordered_map<std::string, std::string> m = {
        // Lowercase Greek
        {"\\alpha",   "\xCE\xB1"}, {"\\beta",    "\xCE\xB2"}, {"\\gamma",   "\xCE\xB3"},
        {"\\delta",   "\xCE\xB4"}, {"\\epsilon", "\xCE\xB5"}, {"\\varepsilon", "\xCE\xB5"},
        {"\\zeta",    "\xCE\xB6"}, {"\\eta",     "\xCE\xB7"}, {"\\theta",   "\xCE\xB8"},
        {"\\vartheta","\xCF\x91"}, {"\\iota",    "\xCE\xB9"}, {"\\kappa",   "\xCE\xBA"},
        {"\\lambda",  "\xCE\xBB"}, {"\\mu",      "\xCE\xBC"}, {"\\nu",      "\xCE\xBD"},
        {"\\xi",      "\xCE\xBE"}, {"\\pi",      "\xCF\x80"}, {"\\varpi",   "\xCF\x96"},
        {"\\rho",     "\xCF\x81"}, {"\\varrho",  "\xCF\xB1"}, {"\\sigma",   "\xCF\x83"},
        {"\\varsigma","\xCF\x82"}, {"\\tau",     "\xCF\x84"}, {"\\upsilon", "\xCF\x85"},
        {"\\phi",     "\xCF\x86"}, {"\\varphi",  "\xCF\x95"}, {"\\chi",     "\xCF\x87"},
        {"\\psi",     "\xCF\x88"}, {"\\omega",   "\xCF\x89"},
        // Uppercase Greek
        {"\\Gamma",   "\xCE\x93"}, {"\\Delta",   "\xCE\x94"}, {"\\Theta",   "\xCE\x98"},
        {"\\Lambda",  "\xCE\x9B"}, {"\\Xi",      "\xCE\x9E"}, {"\\Pi",      "\xCE\xA0"},
        {"\\Sigma",   "\xCE\xA3"}, {"\\Upsilon", "\xCE\xA5"}, {"\\Phi",     "\xCE\xA6"},
        {"\\Psi",     "\xCE\xA8"}, {"\\Omega",   "\xCE\xA9"},
        // Math operators — with thin spaces for readability
        {"\\sum",     " \xE2\x88\x91"},  {"\\prod",    " \xE2\x88\x8F"},
        {"\\int",     "\xE2\x88\xAB"},   {"\\iint",    "\xE2\x88\xAC"},
        {"\\iiint",   "\xE2\x88\xAD"},   {"\\oint",    "\xE2\x88\xAE"},
        {"\\partial", "\xE2\x88\x82"},   {"\\nabla",   "\xE2\x88\x87"},
        {"\\infty",   "\xE2\x88\x9E"},   {"\\propto",  "\xE2\x88\x9D"},
        {"\\approx",  " \xE2\x89\x88 "}, {"\\neq",     " \xE2\x89\xA0 "},
        {"\\leq",     " \xE2\x89\xA4 "}, {"\\geq",     " \xE2\x89\xA5 "},
        {"\\ll",      " \xE2\x89\xAA "}, {"\\gg",      " \xE2\x89\xAB "},
        {"\\equiv",   " \xE2\x89\xA1 "}, {"\\sim",     " \xE2\x88\xBC "},
        {"\\simeq",   " \xE2\x89\x83 "},
        {"\\times",   " \xC3\x97 "},     {"\\cdot",    " \xC2\xB7 "},
        {"\\cdots",   "\xE2\x8B\xAF"},   {"\\ldots",   "\xE2\x80\xA6"},
        {"\\pm",      " \xC2\xB1 "},     {"\\mp",      " \xE2\x88\x93 "},
        {"\\sqrt",    "\xE2\x88\x9A"},   {"\\hbar",    "\xE2\x84\x8F"},
        {"\\ell",     "\xE2\x84\x93"},
        {"\\forall",  "\xE2\x88\x80"},   {"\\exists",  "\xE2\x88\x83"},
        {"\\in",      " \xE2\x88\x88 "}, {"\\notin",   " \xE2\x88\x89 "},
        {"\\subset",  " \xE2\x8A\x82 "}, {"\\supset",  " \xE2\x8A\x83 "},
        {"\\cup",     " \xE2\x88\xAA "}, {"\\cap",     " \xE2\x88\xA9 "},
        {"\\emptyset","\xE2\x88\x85"},
        {"\\dagger",  "\xE2\x80\xA0"},   {"\\ddagger", "\xE2\x80\xA1"},
        {"\\star",    "\xE2\x98\x86"},   {"\\circ",    "\xE2\x88\x98"},
        {"\\bullet",  "\xE2\x80\xA2"},   {"\\otimes",  "\xE2\x8A\x97"},
        {"\\oplus",   "\xE2\x8A\x95"},
        // Arrows
        {"\\rightarrow", " \xE2\x86\x92 "}, {"\\leftarrow",  " \xE2\x86\x90 "},
        {"\\Rightarrow", " \xE2\x87\x92 "}, {"\\Leftarrow",  " \xE2\x87\x90 "},
        {"\\leftrightarrow", " \xE2\x86\x94 "},
        {"\\mapsto",     " \xE2\x86\xA6 "}, {"\\to",         " \xE2\x86\x92 "},
        // Brackets
        {"\\langle",  "\xE2\x9F\xA8"}, {"\\rangle",  "\xE2\x9F\xA9"},
        {"\\lceil",   "\xE2\x8C\x88"}, {"\\rceil",   "\xE2\x8C\x89"},
        {"\\lfloor",  "\xE2\x8C\x8A"}, {"\\rfloor",  "\xE2\x8C\x8B"},
        // Misc
        {"\\&", "&"}, {"\\%", "%"}, {"\\#", "#"},
        {"\\quad", "  "}, {"\\qquad", "    "},
        {"\\,", " "},  {"\\;", " "},  {"\\!", ""},
    };
    return m;
}

// Unicode subscript mapping — single char → subscript string
static const char* subChar(char c) {
    static const std::unordered_map<char, const char*> m = {
        // digits
        {'0',"\xE2\x82\x80"},{'1',"\xE2\x82\x81"},{'2',"\xE2\x82\x82"},
        {'3',"\xE2\x82\x83"},{'4',"\xE2\x82\x84"},{'5',"\xE2\x82\x85"},
        {'6',"\xE2\x82\x86"},{'7',"\xE2\x82\x87"},{'8',"\xE2\x82\x88"},
        {'9',"\xE2\x82\x89"},
        // letters
        {'a',"\xE2\x82\x90"},{'e',"\xE2\x82\x91"},{'h',"\xE2\x82\x95"},
        {'i',"\xE1\xB5\xA2"},{'j',"\xE2\xB1\xBC"},{'k',"\xE2\x82\x96"},
        {'l',"\xE2\x82\x97"},{'m',"\xE2\x82\x98"},{'n',"\xE2\x82\x99"},
        {'o',"\xE2\x82\x92"},{'p',"\xE2\x82\x93"},{'r',"\xE1\xB5\xA3"},
        {'s',"\xE2\x82\x9B"},{'t',"\xE2\x82\x9C"},{'u',"\xE1\xB5\xA4"},
        {'v',"\xE1\xB5\xA5"},{'x',"\xE2\x82\x93"},
        // punctuation
        {'+',"\xE2\x82\x8A"},{'-',"\xE2\x82\x8B"},
        {'(',"\xE2\x82\x8D"},{')',"\xE2\x82\x8E"},
        {'=',"\xE2\x82\x8C"},
    };
    auto it = m.find(c);
    return it != m.end() ? it->second : nullptr;
}

// Unicode superscript mapping
static const char* supChar(char c) {
    static const std::unordered_map<char, const char*> m = {
        // digits
        {'0',"\xC2\xB0"},{'1',"\xC2\xB9"},{'2',"\xC2\xB2"},
        {'3',"\xC2\xB3"},{'4',"\xE2\x81\xB4"},{'5',"\xE2\x81\xB5"},
        {'6',"\xE2\x81\xB6"},{'7',"\xE2\x81\xB7"},{'8',"\xE2\x81\xB8"},
        {'9',"\xE2\x81\xB9"},
        // letters
        {'n',"\xE2\x81\xBF"},{'i',"\xE2\x81\xB1"},
        // punctuation
        {'+',"\xE2\x81\xBA"},{'-',"\xE2\x81\xBB"},
        {'(',"\xE2\x81\xBD"},{')',"\xE2\x81\xBE"},
        {'=',"\xE2\x81\xBC"},
        {'/',"\xE2\x81\x84"},  // fraction slash — works as superscript-ish
    };
    auto it = m.find(c);
    return it != m.end() ? it->second : nullptr;
}

// Common vulgar fractions
static const char* vulgarFraction(const std::string& num, const std::string& den) {
    static const std::unordered_map<std::string, const char*> m = {
        {"1/2",  "\xC2\xBD"},      // ½
        {"1/3",  "\xE2\x85\x93"},  // ⅓
        {"2/3",  "\xE2\x85\x94"},  // ⅔
        {"1/4",  "\xC2\xBC"},      // ¼
        {"3/4",  "\xC2\xBE"},      // ¾
        {"1/5",  "\xE2\x85\x95"},  // ⅕
        {"2/5",  "\xE2\x85\x96"},  // ⅖
        {"3/5",  "\xE2\x85\x97"},  // ⅗
        {"4/5",  "\xE2\x85\x98"},  // ⅘
        {"1/6",  "\xE2\x85\x99"},  // ⅙
        {"5/6",  "\xE2\x85\x9A"},  // ⅚
        {"1/7",  "\xE2\x85\x9B"},  // ⅐
        {"1/8",  "\xE2\x85\x9B"},  // ⅛
        {"3/8",  "\xE2\x85\x9C"},  // ⅜
        {"5/8",  "\xE2\x85\x9D"},  // ⅝
        {"7/8",  "\xE2\x85\x9E"},  // ⅞
    };
    std::string key = num + "/" + den;
    auto it = m.find(key);
    return it != m.end() ? it->second : nullptr;
}

// Find matching closing brace, handling nesting
static size_t findMatchingBrace(const std::string& s, size_t openPos) {
    if (openPos >= s.size() || s[openPos] != '{') return std::string::npos;
    int depth = 1;
    for (size_t i = openPos + 1; i < s.size(); i++) {
        if (s[i] == '{') depth++;
        else if (s[i] == '}') { depth--; if (depth == 0) return i; }
    }
    return std::string::npos;
}

// Try to convert a string to Unicode superscript. Returns empty if any char fails.
static std::string toSuperscript(const std::string& s) {
    std::string result;
    for (char c : s) {
        const char* sc = supChar(c);
        if (sc) result += sc;
        else return "";  // can't map this char
    }
    return result;
}

// Try to convert a string to Unicode subscript. Returns empty if any char fails.
static std::string toSubscript(const std::string& s) {
    std::string result;
    for (char c : s) {
        const char* sc = subChar(c);
        if (sc) result += sc;
        else return "";  // can't map this char
    }
    return result;
}

// Convert a LaTeX math string to Unicode
std::string SciencePanel::latexToUnicode(const std::string& tex) {
    std::string out;
    out.reserve(tex.size() * 2);
    const auto& syms = latexSymbols();

    for (size_t i = 0; i < tex.size(); ) {
        if (tex[i] == '\\' && i + 1 < tex.size()) {
            // ── Decorators that DON'T use combining chars ──────────
            // \vec{x} → x (just the content — vectors clear from context)
            if (tex.substr(i, 4) == "\\vec" && i + 4 < tex.size() && tex[i+4] == '{') {
                size_t close = findMatchingBrace(tex, i + 4);
                if (close != std::string::npos) {
                    out += latexToUnicode(tex.substr(i + 5, close - i - 5));
                    i = close + 1;
                    continue;
                }
            }
            // \hat{x} → x (hat notation dropped — use \hat in text description)
            if (tex.substr(i, 4) == "\\hat" && i + 4 < tex.size() && tex[i+4] == '{') {
                size_t close = findMatchingBrace(tex, i + 4);
                if (close != std::string::npos) {
                    out += latexToUnicode(tex.substr(i + 5, close - i - 5));
                    i = close + 1;
                    continue;
                }
            }
            // \bar{x}, \dot{x}, \ddot{x}, \tilde{x} → just content
            for (const char* dec : {"\\bar", "\\dot"}) {
                size_t dlen = strlen(dec);
                if (tex.substr(i, dlen) == dec && i + dlen < tex.size() && tex[i+dlen] == '{') {
                    size_t close = findMatchingBrace(tex, i + dlen);
                    if (close != std::string::npos) {
                        out += latexToUnicode(tex.substr(i + dlen + 1, close - i - dlen - 1));
                        i = close + 1;
                        goto next_char;
                    }
                }
            }
            if (tex.substr(i, 5) == "\\ddot" && i + 5 < tex.size() && tex[i+5] == '{') {
                size_t close = findMatchingBrace(tex, i + 5);
                if (close != std::string::npos) {
                    out += latexToUnicode(tex.substr(i + 6, close - i - 6));
                    i = close + 1;
                    continue;
                }
            }
            if (tex.substr(i, 6) == "\\tilde" && i + 6 < tex.size() && tex[i+6] == '{') {
                size_t close = findMatchingBrace(tex, i + 6);
                if (close != std::string::npos) {
                    out += latexToUnicode(tex.substr(i + 7, close - i - 7));
                    i = close + 1;
                    continue;
                }
            }

            // ── \sqrt{x} → √(x),  \sqrt[n]{x} → ⁿ√(x) ──────────
            if (tex.substr(i, 5) == "\\sqrt") {
                size_t pos = i + 5;
                std::string nthRoot;
                if (pos < tex.size() && tex[pos] == '[') {
                    size_t closeBracket = tex.find(']', pos + 1);
                    if (closeBracket != std::string::npos) {
                        nthRoot = latexToUnicode(tex.substr(pos + 1, closeBracket - pos - 1));
                        pos = closeBracket + 1;
                    }
                }
                if (pos < tex.size() && tex[pos] == '{') {
                    size_t close = findMatchingBrace(tex, pos);
                    if (close != std::string::npos) {
                        std::string inner = latexToUnicode(tex.substr(pos + 1, close - pos - 1));
                        if (!nthRoot.empty()) {
                            std::string sup = toSuperscript(nthRoot);
                            out += sup.empty() ? nthRoot : sup;
                        }
                        out += "\xE2\x88\x9A"; // √
                        if (inner.size() > 1) out += "(" + inner + ")";
                        else out += inner;
                        i = close + 1;
                        continue;
                    }
                }
            }

            // ── \frac{num}{den} ────────────────────────────────────
            if (tex.substr(i, 5) == "\\frac" && i + 5 < tex.size() && tex[i+5] == '{') {
                size_t close1 = findMatchingBrace(tex, i + 5);
                if (close1 != std::string::npos && close1 + 1 < tex.size() && tex[close1+1] == '{') {
                    size_t close2 = findMatchingBrace(tex, close1 + 1);
                    if (close2 != std::string::npos) {
                        std::string rawNum = tex.substr(i + 6, close1 - i - 6);
                        std::string rawDen = tex.substr(close1 + 2, close2 - close1 - 2);
                        std::string num = latexToUnicode(rawNum);
                        std::string den = latexToUnicode(rawDen);

                        // Try vulgar fraction first (½, ⅓, ¾, etc.)
                        const char* vf = vulgarFraction(num, den);
                        if (vf) {
                            out += vf;
                        } else if (num.size() <= 6 && den.size() <= 6) {
                            out += num + "/" + den;
                        } else {
                            out += "(" + num + ")/(" + den + ")";
                        }
                        i = close2 + 1;
                        continue;
                    }
                }
            }

            // ── Skip formatting commands ───────────────────────────
            if (tex.substr(i, 5) == "\\left") { i += 5; continue; }
            if (tex.substr(i, 6) == "\\right") { i += 6; continue; }

            // \text{...}, \mathrm{...}, \mathbf{...}, \operatorname{...}
            for (const char* cmd : {"\\operatorname", "\\mathbf", "\\mathrm",
                                     "\\mathcal", "\\mathbb", "\\text"}) {
                size_t clen = strlen(cmd);
                if (tex.substr(i, clen) == cmd) {
                    size_t braceStart = tex.find('{', i);
                    if (braceStart != std::string::npos && braceStart <= i + clen + 1) {
                        size_t close = findMatchingBrace(tex, braceStart);
                        if (close != std::string::npos) {
                            out += tex.substr(braceStart + 1, close - braceStart - 1);
                            i = close + 1;
                            goto next_char;
                        }
                    }
                }
            }

            // ── Try to match known symbol (longest match first) ────
            {
                bool found = false;
                for (int len = 16; len >= 2; len--) {
                    if (i + (size_t)len > tex.size()) continue;
                    std::string cmd = tex.substr(i, len);
                    if (len > 2 && i + (size_t)len < tex.size() && std::isalpha(tex[i + len]))
                        continue;
                    auto it = syms.find(cmd);
                    if (it != syms.end()) {
                        out += it->second;
                        i += len;
                        if (i < tex.size() && tex[i] == ' ') i++;
                        found = true;
                        break;
                    }
                }
                if (found) continue;
            }

            // Unknown command: skip backslash
            i++;
            continue;
        }

        // ── Subscript: _{...} or _x ───────────────────────────────
        if (tex[i] == '_' && i + 1 < tex.size()) {
            i++;
            std::string content;
            if (tex[i] == '{') {
                size_t close = findMatchingBrace(tex, i);
                if (close != std::string::npos) {
                    content = tex.substr(i + 1, close - i - 1);
                    i = close + 1;
                } else { content = std::string(1, tex[i]); i++; }
            } else { content = std::string(1, tex[i]); i++; }

            std::string resolved = latexToUnicode(content);
            std::string sub = toSubscript(resolved);
            out += sub.empty() ? ("_(" + resolved + ")") : sub;
            continue;
        }

        // ── Superscript: ^{...} or ^x ─────────────────────────────
        if (tex[i] == '^' && i + 1 < tex.size()) {
            i++;
            std::string content;
            if (tex[i] == '{') {
                size_t close = findMatchingBrace(tex, i);
                if (close != std::string::npos) {
                    content = tex.substr(i + 1, close - i - 1);
                    i = close + 1;
                } else { content = std::string(1, tex[i]); i++; }
            } else { content = std::string(1, tex[i]); i++; }

            std::string resolved = latexToUnicode(content);
            std::string sup = toSuperscript(resolved);
            out += sup.empty() ? ("^(" + resolved + ")") : sup;
            continue;
        }

        // -- → en-dash
        if (tex[i] == '-' && i + 1 < tex.size() && tex[i+1] == '-') {
            out += "\xE2\x80\x93";
            i += 2;
            continue;
        }

        // Strip braces used for grouping
        if (tex[i] == '{' || tex[i] == '}') { i++; continue; }

        // Pass through
        out += tex[i];
        i++;
        continue;

        next_char:;  // goto target for nested loops
    }
    return out;
}

// Process inline $...$ math and LaTeX commands in paragraph text
std::string SciencePanel::convertInlineMath(const std::string& text) {
    std::string out;
    out.reserve(text.size() * 2);

    for (size_t i = 0; i < text.size(); ) {
        if (text[i] == '$') {
            // Find closing $
            size_t close = text.find('$', i + 1);
            if (close != std::string::npos) {
                out += latexToUnicode(text.substr(i + 1, close - i - 1));
                i = close + 1;
                continue;
            }
        }
        // Handle \command outside math mode too (Greek letters in text)
        if (text[i] == '\\' && i + 1 < text.size() && std::isalpha(text[i+1])) {
            const auto& syms = latexSymbols();
            bool found = false;
            for (int len = 14; len >= 2; len--) {
                if (i + len > text.size()) continue;
                std::string cmd = text.substr(i, len);
                if ((size_t)len < text.size() - i && std::isalpha(text[i + len]))
                    continue;
                auto it = syms.find(cmd);
                if (it != syms.end()) {
                    out += it->second;
                    i += len;
                    if (i < text.size() && text[i] == ' ') i++;
                    found = true;
                    break;
                }
            }
            if (found) continue;
        }
        // \& → &
        if (text[i] == '\\' && i + 1 < text.size() && text[i+1] == '&') {
            out += '&'; i += 2; continue;
        }
        // -- → en-dash
        if (text[i] == '-' && i + 1 < text.size() && text[i+1] == '-') {
            out += "\xE2\x80\x93"; i += 2; continue;
        }

        out += text[i];
        i++;
    }
    return out;
}

// =====================================================================
//  LaTeX file parser
// =====================================================================

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Extract content inside braces: \command{content}
static std::string extractBraces(const std::string& line, size_t start) {
    size_t open = line.find('{', start);
    if (open == std::string::npos) return "";
    size_t close = line.find('}', open + 1);
    if (close == std::string::npos) return line.substr(open + 1);
    return line.substr(open + 1, close - open - 1);
}

bool SciencePanel::loadDocument(Document& doc, const std::string& path) {
    // Check if reload needed
    std::error_code ec;
    auto mtime = std::filesystem::last_write_time(path, ec);
    if (ec) return false;
    if (doc.sourcePath == path && mtime == doc.mtime && !doc.blocks.empty())
        return true; // Already loaded, unchanged

    std::ifstream file(path);
    if (!file.is_open()) return false;

    doc.blocks.clear();
    doc.sourcePath = path;
    doc.mtime = mtime;

    std::string line;
    std::string paraAccum;  // Accumulate paragraph lines
    bool inEquation = false;
    std::string eqAccum;
    bool inItemize = false;

    auto flushParagraph = [&]() {
        std::string t = trim(paraAccum);
        if (!t.empty()) {
            doc.blocks.push_back({Block::Paragraph, convertInlineMath(t), ""});
        }
        paraAccum.clear();
    };

    while (std::getline(file, line)) {
        std::string trimmed = trim(line);

        // Equation environment
        if (trimmed == "\\begin{equation}" || trimmed == "\\begin{equation*}") {
            flushParagraph();
            inEquation = true;
            eqAccum.clear();
            continue;
        }
        if (trimmed == "\\end{equation}" || trimmed == "\\end{equation*}") {
            inEquation = false;
            std::string formula = trim(eqAccum);
            if (!formula.empty()) {
                doc.blocks.push_back({Block::Formula, formula, ""}); // Store raw LaTeX
            }
            continue;
        }
        if (inEquation) {
            if (!eqAccum.empty()) eqAccum += " ";
            eqAccum += trimmed;
            continue;
        }

        // Itemize environment
        if (trimmed == "\\begin{itemize}") {
            flushParagraph();
            inItemize = true;
            continue;
        }
        if (trimmed == "\\end{itemize}") {
            inItemize = false;
            continue;
        }
        if (inItemize && trimmed.substr(0, 5) == "\\item") {
            std::string content = trim(trimmed.substr(5));
            doc.blocks.push_back({Block::Bullet, convertInlineMath(content), ""});
            continue;
        }

        // \section{...}
        if (trimmed.substr(0, 8) == "\\section") {
            flushParagraph();
            doc.blocks.push_back({Block::Heading, extractBraces(trimmed, 0), ""});
            continue;
        }

        // \subsection{...}
        if (trimmed.substr(0, 11) == "\\subsection") {
            flushParagraph();
            doc.blocks.push_back({Block::Subheading, extractBraces(trimmed, 0), ""});
            continue;
        }

        // \param{symbol}{description}
        if (trimmed.substr(0, 6) == "\\param") {
            flushParagraph();
            size_t open1 = trimmed.find('{');
            if (open1 != std::string::npos) {
                size_t close1 = trimmed.find('}', open1 + 1);
                if (close1 != std::string::npos) {
                    std::string sym = trimmed.substr(open1 + 1, close1 - open1 - 1);
                    std::string desc = extractBraces(trimmed, close1 + 1);
                    doc.blocks.push_back({Block::Param,
                        latexToUnicode(sym), convertInlineMath(desc)});
                }
            }
            continue;
        }

        // Blank line → end paragraph
        if (trimmed.empty()) {
            flushParagraph();
            continue;
        }

        // Regular text → accumulate paragraph
        if (!paraAccum.empty()) paraAccum += " ";
        paraAccum += trimmed;
    }

    flushParagraph();

    // Set title from first heading
    for (const auto& b : doc.blocks) {
        if (b.type == Block::Heading) { doc.title = b.text; break; }
    }

    return true;
}

// =====================================================================
//  2D Math Layout Engine
// =====================================================================

namespace {

struct MathNode {
    enum Type { Text, Frac, Sub, Sup, SubSup, Sqrt, Row, Space };
    Type type;
    std::string text;
    float spacePx;
    std::vector<MathNode> children;
};

struct MathSize {
    float w, asc, desc;
    float h() const { return asc + desc; }
};

// ── Parser ──────────────────────────────────────────────────────

static MathNode parseMathGroup(const std::string& s, size_t& p);

static MathNode parseBrace(const std::string& s, size_t& p) {
    p++; // skip '{'
    auto n = parseMathGroup(s, p);
    if (p < s.size() && s[p] == '}') p++;
    return n;
}

static bool startsWith(const std::string& s, size_t p, const char* prefix) {
    for (size_t i = 0; prefix[i]; i++) {
        if (p + i >= s.size() || s[p + i] != prefix[i]) return false;
    }
    return true;
}

static MathNode parseMathGroup(const std::string& s, size_t& p) {
    std::vector<MathNode> nodes;
    std::string buf;

    auto flush = [&]() {
        if (!buf.empty()) {
            nodes.push_back({MathNode::Text, buf, 0, {}});
            buf.clear();
        }
    };

    while (p < s.size() && s[p] != '}') {
        char c = s[p];

        if (c == '{') { flush(); nodes.push_back(parseBrace(s, p)); continue; }

        // ── Sub/Superscript ──
        if (c == '_' || c == '^') {
            flush();
            bool isSub = (c == '_');
            p++;
            MathNode content;
            if (p < s.size() && s[p] == '{') content = parseBrace(s, p);
            else if (p < s.size()) { content = {MathNode::Text, std::string(1, s[p]), 0, {}}; p++; }
            else continue;

            MathNode base;
            if (!nodes.empty()) { base = std::move(nodes.back()); nodes.pop_back(); }
            else base = {MathNode::Text, "", 0, {}};

            // Check for other script following
            bool otherIsSub = !isSub;
            char otherChar = otherIsSub ? '_' : '^';
            if (p < s.size() && s[p] == otherChar) {
                p++;
                MathNode content2;
                if (p < s.size() && s[p] == '{') content2 = parseBrace(s, p);
                else if (p < s.size()) { content2 = {MathNode::Text, std::string(1, s[p]), 0, {}}; p++; }
                else { nodes.push_back({isSub ? MathNode::Sub : MathNode::Sup, "", 0, {std::move(base), std::move(content)}}); continue; }

                if (isSub)
                    nodes.push_back({MathNode::SubSup, "", 0, {std::move(base), std::move(content), std::move(content2)}});
                else
                    nodes.push_back({MathNode::SubSup, "", 0, {std::move(base), std::move(content2), std::move(content)}});
            } else {
                nodes.push_back({isSub ? MathNode::Sub : MathNode::Sup, "", 0, {std::move(base), std::move(content)}});
            }
            continue;
        }

        // ── Backslash commands ──
        if (c == '\\') {
            // \frac{num}{den}
            if (startsWith(s, p, "\\frac") && p + 5 < s.size() && s[p+5] == '{') {
                flush(); p += 5;
                auto num = parseBrace(s, p);
                if (p < s.size() && s[p] == '{') {
                    auto den = parseBrace(s, p);
                    nodes.push_back({MathNode::Frac, "", 0, {std::move(num), std::move(den)}});
                }
                continue;
            }
            // \sqrt[n]{x}
            if (startsWith(s, p, "\\sqrt")) {
                flush(); p += 5;
                if (p < s.size() && s[p] == '[') {
                    auto cb = s.find(']', p);
                    if (cb != std::string::npos) p = cb + 1;
                }
                if (p < s.size() && s[p] == '{') {
                    nodes.push_back({MathNode::Sqrt, "", 0, {parseBrace(s, p)}});
                }
                continue;
            }
            // Decorators → just inner content (no combining chars)
            bool handled = false;
            for (const char* d : {"\\tilde", "\\ddot"}) {
                size_t dl = strlen(d);
                if (startsWith(s, p, d) && p + dl < s.size() && s[p+dl] == '{') {
                    flush(); p += dl;
                    nodes.push_back(parseBrace(s, p));
                    handled = true; break;
                }
            }
            if (handled) continue;
            for (const char* d : {"\\vec", "\\hat", "\\bar", "\\dot"}) {
                size_t dl = strlen(d);
                if (startsWith(s, p, d) && p + dl < s.size() && s[p+dl] == '{') {
                    flush(); p += dl;
                    nodes.push_back(parseBrace(s, p));
                    handled = true; break;
                }
            }
            if (handled) continue;
            // \text{}, \mathrm{}, etc → raw text
            for (const char* cmd : {"\\operatorname", "\\mathbf", "\\mathrm", "\\text"}) {
                size_t cl = strlen(cmd);
                if (startsWith(s, p, cmd)) {
                    size_t ob = s.find('{', p + cl);
                    if (ob != std::string::npos && ob <= p + cl + 1) {
                        flush(); p = ob + 1;
                        while (p < s.size() && s[p] != '}') { buf += s[p]; p++; }
                        flush();
                        if (p < s.size()) p++;
                        handled = true; break;
                    }
                }
            }
            if (handled) continue;
            // \left, \right → skip command, pass delimiter
            if (startsWith(s, p, "\\left")) {
                flush(); p += 5;
                if (p < s.size() && !std::isalpha(s[p])) { buf += s[p]; p++; }
                flush(); continue;
            }
            if (startsWith(s, p, "\\right")) {
                flush(); p += 6;
                if (p < s.size() && !std::isalpha(s[p])) { buf += s[p]; p++; }
                flush(); continue;
            }
            // Spacing commands
            if (startsWith(s, p, "\\qquad")) { flush(); nodes.push_back({MathNode::Space, "", 20, {}}); p += 6; continue; }
            if (startsWith(s, p, "\\quad")) { flush(); nodes.push_back({MathNode::Space, "", 12, {}}); p += 5; continue; }
            if (p + 1 < s.size() && s[p+1] == ',') { flush(); nodes.push_back({MathNode::Space, "", 3, {}}); p += 2; continue; }
            if (p + 1 < s.size() && s[p+1] == ';') { flush(); nodes.push_back({MathNode::Space, "", 4, {}}); p += 2; continue; }
            if (p + 1 < s.size() && s[p+1] == '!') { p += 2; continue; }

            // Symbol table lookup
            {
                const auto& syms = latexSymbols();
                bool found = false;
                for (int len = 16; len >= 2; len--) {
                    if (p + (size_t)len > s.size()) continue;
                    std::string cmd = s.substr(p, len);
                    if (len > 2 && p + (size_t)len < s.size() && std::isalpha(s[p + len])) continue;
                    auto it = syms.find(cmd);
                    if (it != syms.end()) {
                        // Strip leading/trailing spaces from symbol value
                        std::string sym = it->second;
                        size_t a = sym.find_first_not_of(' ');
                        size_t b = sym.find_last_not_of(' ');
                        if (a != std::string::npos) sym = sym.substr(a, b - a + 1);

                        bool isOp = (cmd == "\\cdot" || cmd == "\\times" || cmd == "\\pm" ||
                                     cmd == "\\mp" || cmd == "\\approx" || cmd == "\\neq" ||
                                     cmd == "\\leq" || cmd == "\\geq" || cmd == "\\equiv" ||
                                     cmd == "\\sim" || cmd == "\\simeq" || cmd == "\\rightarrow" ||
                                     cmd == "\\to" || cmd == "\\Rightarrow" || cmd == "\\ll" ||
                                     cmd == "\\gg" || cmd == "\\in" || cmd == "\\propto");
                        if (isOp) {
                            flush();
                            nodes.push_back({MathNode::Space, "", 4, {}});
                            nodes.push_back({MathNode::Text, sym, 0, {}});
                            nodes.push_back({MathNode::Space, "", 4, {}});
                        } else {
                            buf += sym;
                        }
                        p += len;
                        if (p < s.size() && s[p] == ' ') p++;
                        found = true;
                        break;
                    }
                }
                if (found) continue;
            }
            p++; // skip unknown backslash
            continue;
        }

        // Space → thin space
        if (c == ' ') { flush(); nodes.push_back({MathNode::Space, "", 3, {}}); p++; continue; }

        // = with medium space around it
        if (c == '=') {
            flush();
            nodes.push_back({MathNode::Space, "", 5, {}});
            nodes.push_back({MathNode::Text, "=", 0, {}});
            nodes.push_back({MathNode::Space, "", 5, {}});
            p++; continue;
        }

        // Regular character
        buf += c;
        p++;
    }

    flush();
    if (nodes.empty()) return {MathNode::Text, "", 0, {}};
    if (nodes.size() == 1) return std::move(nodes[0]);
    return {MathNode::Row, "", 0, std::move(nodes)};
}

static MathNode parseMath(const std::string& latex) {
    size_t p = 0;
    return parseMathGroup(latex, p);
}

// ── Measurement ─────────────────────────────────────────────────

static float fontAsc(ImFont* f) { return f->LegacySize * 0.76f; }
static float fontDesc(ImFont* f) { return f->LegacySize * 0.24f; }

static MathSize msize(const MathNode& n, ImFont* f, ImFont* sf) {
    switch (n.type) {
    case MathNode::Text: {
        if (n.text.empty()) return {0, fontAsc(f), fontDesc(f)};
        auto sz = f->CalcTextSizeA(f->LegacySize, FLT_MAX, 0, n.text.c_str());
        return {sz.x, fontAsc(f), fontDesc(f)};
    }
    case MathNode::Space:
        return {n.spacePx, 0, 0};
    case MathNode::Row: {
        MathSize m = {0, 0, 0};
        for (auto& c : n.children) {
            auto cm = msize(c, f, sf);
            m.w += cm.w;
            m.asc = std::max(m.asc, cm.asc);
            m.desc = std::max(m.desc, cm.desc);
        }
        return m;
    }
    case MathNode::Frac: {
        auto nm = msize(n.children[0], sf, sf);
        auto dm = msize(n.children[1], sf, sf);
        float pad = 3.0f, gap = 2.5f;
        float axis = f->LegacySize * 0.3f;
        float w = std::max(nm.w, dm.w) + pad * 2;
        return {w, nm.h() + gap + axis, dm.h() + gap + f->LegacySize * 0.2f};
    }
    case MathNode::Sub: {
        auto bm = msize(n.children[0], f, sf);
        auto sm = msize(n.children[1], sf, sf);
        float drop = f->LegacySize * 0.25f;
        return {bm.w + sm.w + 1, bm.asc, std::max(bm.desc, drop + sm.h())};
    }
    case MathNode::Sup: {
        auto bm = msize(n.children[0], f, sf);
        auto sm = msize(n.children[1], sf, sf);
        float rise = f->LegacySize * 0.38f;
        return {bm.w + sm.w + 1, std::max(bm.asc, rise + sm.h()), bm.desc};
    }
    case MathNode::SubSup: {
        auto bm = msize(n.children[0], f, sf);
        auto subm = msize(n.children[1], sf, sf);
        auto supm = msize(n.children[2], sf, sf);
        float scriptW = std::max(subm.w, supm.w);
        float drop = f->LegacySize * 0.25f;
        float rise = f->LegacySize * 0.38f;
        return {bm.w + scriptW + 1,
                std::max(bm.asc, rise + supm.h()),
                std::max(bm.desc, drop + subm.h())};
    }
    case MathNode::Sqrt: {
        auto cm = msize(n.children[0], f, sf);
        float radW = f->LegacySize * 0.55f;
        return {radW + cm.w + 3, cm.asc + 4, cm.desc};
    }
    }
    return {0, 0, 0};
}

// ── Rendering ───────────────────────────────────────────────────

static void mdraw(ImDrawList* dl, const MathNode& n, float x, float bl,
                  ImFont* f, ImFont* sf, ImU32 tc, ImU32 lc) {
    switch (n.type) {
    case MathNode::Text:
        if (!n.text.empty())
            dl->AddText(f, f->LegacySize, ImVec2(x, bl - fontAsc(f)), tc, n.text.c_str());
        break;
    case MathNode::Space:
        break;
    case MathNode::Row: {
        float cx = x;
        for (auto& c : n.children) {
            mdraw(dl, c, cx, bl, f, sf, tc, lc);
            cx += msize(c, f, sf).w;
        }
        break;
    }
    case MathNode::Frac: {
        auto nm = msize(n.children[0], sf, sf);
        auto dm = msize(n.children[1], sf, sf);
        float pad = 3.0f, gap = 2.5f;
        float barW = std::max(nm.w, dm.w) + pad * 2;
        float axis = f->LegacySize * 0.3f;
        float barY = bl - axis;
        // Numerator centered above bar
        float numBl = barY - gap - nm.desc;
        mdraw(dl, n.children[0], x + (barW - nm.w) / 2, numBl, sf, sf, tc, lc);
        // Fraction bar
        dl->AddLine(ImVec2(x, barY), ImVec2(x + barW, barY), lc, 1.0f);
        // Denominator centered below bar
        float denBl = barY + 1 + gap + dm.asc;
        mdraw(dl, n.children[1], x + (barW - dm.w) / 2, denBl, sf, sf, tc, lc);
        break;
    }
    case MathNode::Sub: {
        auto bm = msize(n.children[0], f, sf);
        mdraw(dl, n.children[0], x, bl, f, sf, tc, lc);
        float drop = f->LegacySize * 0.25f;
        mdraw(dl, n.children[1], x + bm.w + 1, bl + drop, sf, sf, tc, lc);
        break;
    }
    case MathNode::Sup: {
        auto bm = msize(n.children[0], f, sf);
        mdraw(dl, n.children[0], x, bl, f, sf, tc, lc);
        float rise = f->LegacySize * 0.38f;
        mdraw(dl, n.children[1], x + bm.w + 1, bl - rise, sf, sf, tc, lc);
        break;
    }
    case MathNode::SubSup: {
        auto bm = msize(n.children[0], f, sf);
        mdraw(dl, n.children[0], x, bl, f, sf, tc, lc);
        float sx = x + bm.w + 1;
        mdraw(dl, n.children[1], sx, bl + f->LegacySize * 0.25f, sf, sf, tc, lc);
        mdraw(dl, n.children[2], sx, bl - f->LegacySize * 0.38f, sf, sf, tc, lc);
        break;
    }
    case MathNode::Sqrt: {
        auto cm = msize(n.children[0], f, sf);
        float radW = f->LegacySize * 0.55f;
        float topY = bl - cm.asc - 2;
        float botY = bl + cm.desc;
        // Radical sign
        dl->AddLine(ImVec2(x + 2, bl - cm.asc * 0.3f), ImVec2(x + radW * 0.4f, botY), lc, 1.0f);
        dl->AddLine(ImVec2(x + radW * 0.4f, botY), ImVec2(x + radW, topY), lc, 1.5f);
        dl->AddLine(ImVec2(x + radW, topY), ImVec2(x + radW + cm.w + 2, topY), lc, 1.0f);
        // Content
        mdraw(dl, n.children[0], x + radW + 1, bl, f, sf, tc, lc);
        break;
    }
    }
}

} // anonymous namespace

// =====================================================================
//  ImGui rendering helpers
// =====================================================================

void SciencePanel::heading(const char* text) {
    spacer(6.0f);
    if (m_fontHeading) ImGui::PushFont(m_fontHeading);
    ImGui::TextColored(ImVec4(0.70f, 0.85f, 1.0f, 1.0f), "%s", text);
    if (m_fontHeading) ImGui::PopFont();

    // Gradient underline: bright on left, fading to transparent
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilledMultiColor(
        ImVec2(p.x, p.y), ImVec2(p.x + w, p.y + 2.0f),
        IM_COL32(80, 140, 255, 180), IM_COL32(80, 140, 255, 0),
        IM_COL32(80, 140, 255, 0),   IM_COL32(80, 140, 255, 180));
    spacer(8.0f);
}

void SciencePanel::subheading(const char* text) {
    spacer(2.0f);
    if (m_fontHeading) ImGui::PushFont(m_fontHeading);
    ImGui::TextColored(ImVec4(0.80f, 0.90f, 1.0f, 1.0f), "%s", text);
    if (m_fontHeading) ImGui::PopFont();
    spacer(2.0f);
}

void SciencePanel::paragraph(const char* text) {
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX());
    ImGui::TextColored(ImVec4(0.88f, 0.88f, 0.92f, 1.0f), "%s", text);
    ImGui::PopTextWrapPos();
    spacer(4.0f);
}

void SciencePanel::formula(const char* rawLatex) {
    ImFont* mainFont = m_fontFormula;
    ImFont* smallFont = m_fontFormulaSmall;
    if (!mainFont || !smallFont) return;

    // Parse LaTeX → expression tree → measure
    MathNode tree = parseMath(rawLatex);
    MathSize ms = msize(tree, mainFont, smallFont);

    float padX = mainFont->LegacySize * 0.6f;
    float padY = mainFont->LegacySize * 0.45f;
    float accentW = 3.0f;
    float contentW = ImGui::GetContentRegionAvail().x;

    ImVec2 cursor = ImGui::GetCursorScreenPos();
    float boxH = ms.h() + padY * 2;

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Full-width dark background
    dl->AddRectFilled(
        ImVec2(cursor.x, cursor.y),
        ImVec2(cursor.x + contentW, cursor.y + boxH),
        IM_COL32(14, 18, 32, 220), 4.0f);
    // Subtle border
    dl->AddRect(
        ImVec2(cursor.x, cursor.y),
        ImVec2(cursor.x + contentW, cursor.y + boxH),
        IM_COL32(60, 80, 140, 60), 4.0f, 0, 1.0f);
    // Left accent bar
    dl->AddRectFilled(
        ImVec2(cursor.x, cursor.y + 2),
        ImVec2(cursor.x + accentW, cursor.y + boxH - 2),
        IM_COL32(80, 140, 255, 180), 2.0f);

    // Render formula with 2D layout engine
    float formulaX = cursor.x + accentW + padX;
    float baseline = cursor.y + padY + ms.asc;
    ImU32 textCol = IM_COL32(200, 225, 255, 255);
    ImU32 lineCol = IM_COL32(140, 175, 230, 220);

    mdraw(dl, tree, formulaX, baseline, mainFont, smallFont, textCol, lineCol);

    // Advance ImGui cursor past the box
    ImGui::SetCursorScreenPos(ImVec2(cursor.x, cursor.y + boxH));
    spacer(6.0f);
}

void SciencePanel::bullet(const char* text) {
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX());
    ImVec2 p = ImGui::GetCursorScreenPos();
    float fontH = ImGui::GetFontSize();
    ImGui::GetWindowDrawList()->AddCircleFilled(
        ImVec2(p.x + 4.0f, p.y + fontH * 0.45f), 2.5f,
        IM_COL32(90, 155, 255, 200));
    ImGui::Indent(14.0f);
    ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.90f, 1.0f), "%s", text);
    ImGui::Unindent(14.0f);
    ImGui::PopTextWrapPos();
}

void SciencePanel::paramRow(const char* symbol, const char* description) {
    ImVec2 cursor = ImGui::GetCursorScreenPos();
    float rowW = ImGui::GetContentRegionAvail().x;
    float fontSize = ImGui::GetFontSize();

    // Subtle alternating row background
    ImGui::GetWindowDrawList()->AddRectFilled(
        ImVec2(cursor.x, cursor.y),
        ImVec2(cursor.x + rowW, cursor.y + fontSize + 6.0f),
        IM_COL32(25, 30, 50, 100), 3.0f);

    ImGui::Indent(8.0f);
    if (m_fontFormula) ImGui::PushFont(m_fontFormula);
    ImGui::TextColored(ImVec4(0.75f, 0.88f, 1.0f, 1.0f), "%s", symbol);
    if (m_fontFormula) ImGui::PopFont();
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.50f, 0.55f, 0.65f, 1.0f), " \xE2\x80\x94 "); // em-dash
    ImGui::SameLine();
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX());
    ImGui::TextColored(ImVec4(0.78f, 0.78f, 0.84f, 1.0f), "%s", description);
    ImGui::PopTextWrapPos();
    ImGui::Unindent(8.0f);
    spacer(2.0f);
}

void SciencePanel::spacer(float h) {
    ImGui::Dummy(ImVec2(0.0f, h));
}

// =====================================================================
//  Render a parsed document
// =====================================================================

void SciencePanel::renderDocument(Document& doc) {
    if (doc.blocks.empty()) return;

    std::string winTitle = doc.title.empty() ? "Science" : doc.title;
    winTitle += "###SciencePanel";

    ImGui::SetNextWindowPos(ImVec2(360, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, 680), ImGuiCond_FirstUseEver);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.08f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));

    // Use the zoom-appropriate body font for the window
    ImFont* zBody    = m_fontBodyZoom[m_zoomLevel];
    ImFont* zHeading = m_fontHeadingZoom[m_zoomLevel];
    ImFont* zFormula = m_fontFormulaZoom[m_zoomLevel];
    if (zBody) ImGui::PushFont(zBody);

    bool open = true;
    ImGui::Begin(winTitle.c_str(), &open, ImGuiWindowFlags_NoFocusOnAppearing);

    if (!open) {
        ImGui::End();
        if (zBody) ImGui::PopFont();
        ImGui::PopStyleColor(2);
        return;
    }

    // Zoom buttons — discrete levels
    static const char* zoomLabels[] = {"75%", "100%", "125%", "150%", "175%"};
    if (ImGui::SmallButton("-##zoom") && m_zoomLevel > 0) { m_zoomLevel--; }
    ImGui::SameLine();
    ImGui::Text("%s", zoomLabels[m_zoomLevel]);
    ImGui::SameLine();
    if (ImGui::SmallButton("+##zoom") && m_zoomLevel < ZOOM_LEVELS - 1) { m_zoomLevel++; }
    ImGui::Separator();

    // Temporarily set the zoom-level fonts for the render helpers
    ImFont* savedBody    = m_fontBody;
    ImFont* savedHeading = m_fontHeading;
    ImFont* savedFormula = m_fontFormula;
    ImFont* savedFormulaSmall = m_fontFormulaSmall;
    m_fontBody    = zBody;
    m_fontHeading = zHeading;
    m_fontFormula = zFormula;
    m_fontFormulaSmall = m_fontFormulaZoom[std::max(0, m_zoomLevel - 1)];

    for (const auto& b : doc.blocks) {
        switch (b.type) {
            case Block::Heading:    heading(b.text.c_str()); break;
            case Block::Subheading: subheading(b.text.c_str()); break;
            case Block::Paragraph:  paragraph(b.text.c_str()); break;
            case Block::Formula:    formula(b.text.c_str()); break;
            case Block::Bullet:     bullet(b.text.c_str()); break;
            case Block::Param:      paramRow(b.text.c_str(), b.extra.c_str()); break;
        }
    }

    m_fontBody    = savedBody;
    m_fontHeading = savedHeading;
    m_fontFormula = savedFormula;
    m_fontFormulaSmall = savedFormulaSmall;

    ImGui::End();
    if (zBody) ImGui::PopFont();
    ImGui::PopStyleColor(2);
}

// =====================================================================
//  Dispatch
// =====================================================================

void SciencePanel::render(ComputeBackend backend) {
    // Periodically check for file changes (~every 60 frames)
    m_reloadCounter++;
    bool doReload = (m_reloadCounter % 60 == 0);

    Document* doc = nullptr;
    std::string filename;

    switch (backend) {
        case ComputeBackend::BruteForce:
        case ComputeBackend::BarnesHut:
            doc = &m_nbody; filename = "nbody.tex"; break;
        case ComputeBackend::PM:
            doc = &m_pm; filename = "pm.tex"; break;
        case ComputeBackend::Inflation:
            doc = &m_inflation; filename = "inflation.tex"; break;
        case ComputeBackend::CDT2D:
            doc = &m_cdt2d; filename = "cdt2d.tex"; break;
        case ComputeBackend::CDT3D:
            doc = &m_cdt3d; filename = "cdt3d.tex"; break;
        default:
            return;
    }

    // Try per-simulation papers/index.tex first, fall back to legacy naming
    std::string path = m_papersDir + "/index.tex";
    if (!std::filesystem::exists(path)) {
        path = m_papersDir + "/" + filename;
    }

    if (doc->blocks.empty() || doReload) {
        loadDocument(*doc, path);
    }

    renderDocument(*doc);
}

} // namespace cosmico
