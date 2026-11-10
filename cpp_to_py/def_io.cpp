// DEF reader and writer in C++ (no GPU): the same result as the Python code in utils/io_parser.py, which stays as
// the fallback and as the reference this file is tested against. One pass over the memory-mapped file.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using sv = std::string_view;
using Tokens = std::vector<sv>;

struct File {                                    // read-only memory map
    const char* p = nullptr;
    size_t n = 0;
    explicit File(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        struct stat st;
        bool ok = fd >= 0 && fstat(fd, &st) == 0;
        void* m = ok && st.st_size ? mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0) : nullptr;
        if (fd >= 0) close(fd);                  // the mapping stays valid after the descriptor is closed
        if (!ok || m == MAP_FAILED) throw std::runtime_error("cannot read " + path);
        p = (const char*)m;
        n = st.st_size;
    }
    ~File() { if (n) munmap((void*)p, n); }
};

struct Lexer {                                   // tokens are separated by white space; ';' is always its own token
    const char *begin, *p, *end;
    static bool space(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
    sv next() {
        for (;;) {
            while (p < end && space(*p)) p++;
            if (p < end && *p == '#' && (p == begin || p[-1] == '\n')) while (p < end && *p != '\n') p++;   // comment line
            else break;
        }
        if (p == end) return {};
        if (*p == ';') return sv(p++, 1);
        const char* s = p;
        while (p < end && !space(*p) && *p != ';') p++;
        return sv(s, p - s);
    }
    void statement(Tokens& t) {                  // the tokens up to the next ';' (not included)
        t.clear();
        for (sv k = next(); !k.empty() && k != ";"; k = next()) t.push_back(k);
    }
    void skip_statement() { for (sv k = next(); !k.empty() && k != ";"; k = next()); }
    void skip_section(sv name) {                 // up to "END <name>"
        for (sv prev, k = next(); !k.empty(); prev = k, k = next()) if (prev == "END" && k == name) return;
    }
};

static long num(sv s) { return std::strtol(std::string(s).c_str(), nullptr, 10); }
static bool placement(sv s) { return s == "PLACED" || s == "FIXED" || s == "COVER"; }
static py::str str(sv s) { return py::str(s.data(), s.size()); }
static py::list strings(const Tokens& v) { py::list l; for (sv s : v) l.append(str(s)); return l; }
template <class T> static py::array_t<T> array(const std::vector<T>& v, ssize_t cols = 0) {
    py::array_t<T> a(cols ? std::vector<ssize_t>{(ssize_t)v.size() / cols, cols} : std::vector<ssize_t>{(ssize_t)v.size()});
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

static py::dict parse_def(const std::string& path) {
    File f(path);
    Lexer lx{f.p, f.p, f.p + f.n};
    Tokens t, comp_name, comp_macro, comp_orient, io_name, net_name, row_site, pin_names{sv()}, macro_names;
    sv design;
    long dbu = 1000, soft = 0;
    std::vector<long> die(4, 0), rows, comp_xy, comp_stmt, net_start{0}, pin_comp, pin_name, blockages, comp_macro_id;
    std::vector<bool> fixed, placed;
    std::vector<double> io_xy;
    std::unordered_map<sv, long> comp_id, io_id, name_id{{sv(), 0}}, macro_id;
    bool have_design = false;
    for (sv k = lx.next(); !k.empty(); k = lx.next()) {
        if (k == "DESIGN" && !have_design) { design = lx.next(); have_design = true; lx.skip_statement(); }
        else if (k == "UNITS") {
            lx.statement(t);
            for (size_t i = 0; i + 1 < t.size(); i++) if (t[i] == "MICRONS") dbu = num(t[i + 1]);
        } else if (k == "DIEAREA") {
            lx.statement(t);
            std::vector<long> v;
            for (sv s : t) if (s != "(" && s != ")") v.push_back(num(s));
            for (size_t i = 0; i < v.size(); i++) {
                long &lo = die[i % 2], &hi = die[2 + i % 2];
                if (i < 2) lo = hi = v[i]; else { lo = std::min(lo, v[i]); hi = std::max(hi, v[i]); }
            }
        } else if (k == "ROW") {                 // ROW name site x y orient [DO nx BY ny [STEP sx sy]]
            lx.statement(t);
            if (t.size() < 5) continue;
            row_site.push_back(t[1]);
            long r[6] = {num(t[2]), num(t[3]), 1, 1, 0, 0};
            if (t.size() >= 9 && t[5] == "DO") { r[2] = num(t[6]); r[3] = num(t[8]); }
            if (t.size() >= 12 && t[9] == "STEP") { r[4] = num(t[10]); r[5] = num(t[11]); }
            rows.insert(rows.end(), r, r + 6);
        } else if (k == "COMPONENTS") {
            lx.skip_statement();
            for (k = lx.next(); !k.empty() && k != "END"; k = lx.next()) {
                if (k != "-") continue;
                long begin = k.data() - f.p;
                lx.statement(t);                 // name macro ...
                if (t.size() < 2) continue;
                long x = 0, y = 0;
                sv status, orient = "N";
                for (size_t i = 2; i + 6 < t.size(); i++)
                    if (t[i] == "+" && placement(t[i + 1])) { status = t[i + 1]; x = num(t[i + 3]); y = num(t[i + 4]); orient = t[i + 6]; break; }
                comp_id.emplace(t[0], (long)comp_name.size());
                comp_name.push_back(t[0]);
                auto m = macro_id.emplace(t[1], (long)macro_names.size());
                if (m.second) macro_names.push_back(t[1]);
                comp_macro_id.push_back(m.first->second);
                fixed.push_back(status == "FIXED" || status == "COVER");
                placed.push_back(!status.empty());
                comp_xy.push_back(x); comp_xy.push_back(y);
                comp_orient.push_back(orient);
                comp_stmt.push_back(begin); comp_stmt.push_back(lx.p - f.p);
            }
            lx.next();
        } else if (k == "PINS") {
            lx.skip_statement();
            for (k = lx.next(); !k.empty() && k != "END"; k = lx.next()) {
                if (k != "-") continue;
                lx.statement(t);                 // name + NET net ... + LAYER l ( x y ) ( x y ) ... + FIXED ( x y ) orient
                if (t.empty()) continue;
                bool has_rect = false, has_place = false;
                double cx = 0, cy = 0, px = 0, py_ = 0;
                sv orient;
                for (size_t i = 1; i < t.size(); i++) {
                    if (t[i] == "LAYER" && !has_rect) {
                        size_t j = i;
                        while (j < t.size() && t[j] != "(") j++;
                        if (j + 6 < t.size()) { cx = (num(t[j + 1]) + num(t[j + 5])) / 2.0; cy = (num(t[j + 2]) + num(t[j + 6])) / 2.0; has_rect = true; }
                    } else if (placement(t[i]) && t[i - 1] == "+" && !has_place && i + 5 < t.size()) {
                        px = num(t[i + 2]); py_ = num(t[i + 3]); orient = t[i + 5]; has_place = true;
                    }
                }
                io_id.emplace(t[0], (long)io_name.size());
                io_name.push_back(t[0]);
                if (!has_place) { io_xy.push_back(NAN); io_xy.push_back(NAN); continue; }
                double ox = cx, oy = cy;             // the shape centre in the pin's orientation
                if (orient == "S") { ox = -cx; oy = -cy; } else if (orient == "W") { ox = -cy; oy = cx; }
                else if (orient == "E") { ox = cy; oy = -cx; } else if (orient == "FN") { ox = -cx; }
                else if (orient == "FS") { oy = -cy; } else if (orient == "FW") { ox = cy; oy = cx; }
                else if (orient == "FE") { ox = -cy; oy = -cx; }
                io_xy.push_back(px + ox); io_xy.push_back(py_ + oy);
            }
            lx.next();
        } else if (k == "NETS") {
            lx.skip_statement();
            for (k = lx.next(); !k.empty() && k != "END"; k = lx.next()) {
                if (k != "-") continue;
                net_name.push_back(lx.next());
                for (k = lx.next(); k == "("; k = lx.next()) {      // ( component pin ) or ( PIN name ), extras ignored
                    sv a = lx.next(), b = lx.next();
                    auto& ids = a == "PIN" ? io_id : comp_id;
                    auto it = ids.find(a == "PIN" ? b : a);
                    if (it == ids.end()) throw std::runtime_error("net " + std::string(net_name.back()) + ": unknown " + std::string(a) + " " + std::string(b));
                    if (a == "PIN") { pin_comp.push_back(-1 - it->second); pin_name.push_back(0); }
                    else {
                        auto n = name_id.emplace(b, (long)pin_names.size());
                        if (n.second) pin_names.push_back(b);
                        pin_comp.push_back(it->second); pin_name.push_back(n.first->second);
                    }
                    while (!k.empty() && k != ")") k = lx.next();
                }
                net_start.push_back((long)pin_comp.size());
                if (k != ";") lx.skip_statement();
            }
            lx.next();
        } else if (k == "BLOCKAGES") {
            lx.skip_statement();
            for (k = lx.next(); !k.empty() && k != "END"; k = lx.next()) {
                if (k != "-") continue;
                lx.statement(t);
                if (t.empty() || t[0] != "PLACEMENT") continue;
                bool is_soft = false;
                for (sv s : t) is_soft |= s == "SOFT" || s == "PARTIAL";
                if (is_soft) { soft++; continue; }
                for (size_t i = 1; i + 8 < t.size(); i++)
                    if (t[i] == "RECT") for (int j : {2, 3, 6, 7}) blockages.push_back(num(t[i + j]));
            }
            lx.next();
        } else if (k == "PROPERTYDEFINITIONS" || k == "VIAS" || k == "NONDEFAULTRULES" || k == "REGIONS" ||
                   k == "SPECIALNETS" || k == "GROUPS" || k == "FILLS" || k == "SCANCHAINS" || k == "STYLES" ||
                   k == "SLOTS" || k == "PINPROPERTIES") lx.skip_section(k);
        else if (k != ";") lx.skip_statement();
    }
    std::vector<uint8_t> fixed8(fixed.begin(), fixed.end()), placed8(placed.begin(), placed.end());
    py::dict d;
    d["design"] = str(design); d["dbu"] = dbu; d["die"] = py::make_tuple(die[0], die[1], die[2], die[3]);
    d["row_site"] = strings(row_site); d["rows"] = array(rows, 6);
    d["comp_name"] = strings(comp_name); d["macro_names"] = strings(macro_names); d["comp_macro_id"] = array(comp_macro_id);
    d["comp_fixed"] = array(fixed8); d["comp_placed"] = array(placed8); d["comp_xy"] = array(comp_xy, 2);
    d["comp_orient"] = strings(comp_orient); d["comp_stmt"] = array(comp_stmt, 2);
    d["io_name"] = strings(io_name); d["io_xy"] = array(io_xy, 2);
    d["net_name"] = strings(net_name); d["net_start"] = array(net_start); d["pin_comp"] = array(pin_comp);
    d["pin_name"] = array(pin_name); d["pin_names"] = strings(pin_names);
    d["blockages"] = array(blockages, 4); d["soft_blockages"] = soft;
    return d;
}

// Copy `src` to `dst` with the components index[k] PLACED at xy[k] (lower left). stmt[c] = [begin, end) of the
// statement of component c in src, as parse_def returned it. Everything else is copied byte for byte.
static void write_def(const std::string& src, const std::string& dst, py::array_t<long> stmt, py::array_t<long> index,
                      py::array_t<long> xy, const std::string& orient) {
    File f(src);
    auto s = stmt.unchecked<2>();
    auto idx = index.unchecked<1>();
    auto p = xy.unchecked<2>();
    if (s.shape(1) != 2 || p.shape(0) != idx.shape(0) || p.shape(1) != 2) throw std::runtime_error("write_def: bad array shapes");
    std::vector<long> where(s.shape(0), -1);
    for (ssize_t k = 0; k < idx.shape(0); k++) {
        if (idx(k) < 0 || idx(k) >= s.shape(0)) throw std::runtime_error("write_def: component index out of range");
        where[idx(k)] = k;
    }
    for (ssize_t c = 0; c < s.shape(0); c++)
        if (s(c, 0) < 0 || s(c, 0) > s(c, 1) || s(c, 1) > (long)f.n) throw std::runtime_error("write_def: statement range outside " + src);
    std::vector<char> buffer(1 << 22);           // declared before the file: it must outlive it
    std::unique_ptr<FILE, int (*)(FILE*)> file(std::fopen(dst.c_str(), "wb"), std::fclose);    // closed on every path
    FILE* out = file.get();
    if (!out) throw std::runtime_error("cannot write " + dst);
    std::setvbuf(out, buffer.data(), _IOFBF, buffer.size());
    Tokens t;
    long cur = 0;
    for (ssize_t c = 0; c < s.shape(0); c++) {
        if (where[c] < 0) continue;
        std::fwrite(f.p + cur, 1, s(c, 0) - cur, out);
        Lexer lx{f.p, f.p + s(c, 0), f.p + s(c, 1)};
        lx.statement(t);                         // - name macro ...: drop the old placement status, keep the rest
        for (size_t i = 0; i < t.size();) {
            if (i >= 3 && t[i] == "+" && i + 1 < t.size() && t[i + 1] == "UNPLACED") i += 2;
            else if (i >= 3 && t[i] == "+" && i + 1 < t.size() && placement(t[i + 1])) i += 7;
            else { std::fwrite(t[i].data(), 1, t[i].size(), out); std::fputc(' ', out); i++; }
        }
        std::fprintf(out, "+ PLACED ( %ld %ld ) %s ;", p(where[c], 0), p(where[c], 1), orient.c_str());
        cur = s(c, 1);
    }
    std::fwrite(f.p + cur, 1, f.n - cur, out);
    bool failed = std::ferror(out);
    if (std::fclose(file.release()) != 0 || failed) throw std::runtime_error("writing " + dst + " failed (disk full?)");
}

PYBIND11_MODULE(def_io, m) {
    m.def("parse_def", &parse_def, "read a DEF file into arrays");
    m.def("write_def", &write_def, "copy a DEF file with some components placed");
}
