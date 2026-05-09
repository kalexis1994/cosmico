#pragma once
#include <string>
#include <vector>
#include <filesystem>

struct ImFont;

namespace cosmico {

enum class ComputeBackend;

class SciencePanel {
public:
    static constexpr int ZOOM_LEVELS = 5;

    void init(ImFont* body, ImFont* heading, ImFont* formula,
              ImFont* bodyZoom[], ImFont* headingZoom[], ImFont* formulaZoom[],
              const std::string& papersDir);
    void setPapersDir(const std::string& dir);
    void render(ComputeBackend backend);

private:
    // ── Document model ───────────────────────────────────────────
    struct Block {
        enum Type { Heading, Subheading, Paragraph, Formula, Bullet, Param };
        Type type;
        std::string text;
        std::string extra;  // Param: description
    };

    struct Document {
        std::string title;
        std::vector<Block> blocks;
        std::filesystem::file_time_type mtime{};
        std::string sourcePath;
    };

    // ── LaTeX parser ─────────────────────────────────────────────
    bool loadDocument(Document& doc, const std::string& path);
    static std::string latexToUnicode(const std::string& tex);
    static std::string convertInlineMath(const std::string& text);

    // ── Rendering ────────────────────────────────────────────────
    void renderDocument(Document& doc);
    void heading(const char* text);
    void subheading(const char* text);
    void paragraph(const char* text);
    void formula(const char* text);
    void bullet(const char* text);
    void paramRow(const char* symbol, const char* description);
    void spacer(float h = 6.0f);

    // ── State ────────────────────────────────────────────────────
    ImFont* m_fontBody    = nullptr;
    ImFont* m_fontHeading = nullptr;
    ImFont* m_fontFormula = nullptr;
    ImFont* m_fontFormulaSmall = nullptr; // For subscripts/superscripts (one zoom level down)

    // Discrete zoom fonts
    ImFont* m_fontBodyZoom[ZOOM_LEVELS]    = {};
    ImFont* m_fontHeadingZoom[ZOOM_LEVELS] = {};
    ImFont* m_fontFormulaZoom[ZOOM_LEVELS] = {};

    std::string m_papersDir;
    int m_zoomLevel = 1; // index into zoom arrays (0=75%, 1=100%, 2=125%, 3=150%, 4=175%)
    int m_reloadCounter = 0;

    Document m_nbody;
    Document m_pm;
    Document m_inflation;
    Document m_cdt2d;
    Document m_cdt3d;
};

} // namespace cosmico
