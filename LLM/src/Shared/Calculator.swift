import Foundation

// A small scientific expression evaluator with variable memory -- the engine
// behind the always-available `calculator` tool. Pure Swift: a hand-written
// lexer and recursive-descent parser over a FIXED operator / function table, so
// a model-authored (hence untrusted) expression never reaches NSExpression,
// whose format grammar can invoke arbitrary Objective-C selectors. Variables
// assigned with `x = ...` persist across calls within a session; reset() clears
// them (New Chat). An actor, so its mutable context is safe to hold inside a
// Sendable tool runner.
//
// Values are COMPLEX (the constant `i`), with principal-branch exp / ln /
// sqrt / pow, so the Euler-formula walk a model loves to take -- exp(i*pi),
// i^i, sqrt(-1), ln(-1), cos(1)+i*sin(1) -- computes instead of erroring.
// Real inputs keep im == 0 exactly through + - * / and the real-fast-path
// functions, so every purely real expression formats exactly as before.
public actor Calculator {
    private var context: [String: Cx] = [:]

    private static let constants: [String: Cx] = [
        "pi": Cx(.pi), "e": Cx(M_E), "tau": Cx(.pi * 2), "i": Cx(0, 1),
    ]

    public init() {}

    public func reset() { context.removeAll() }

    // Evaluate an assignment ("x = expr") or a bare expression ("x * 3").
    // Returns the formatted number, or an "error: ..." string the model can
    // react to. An assignment stores the value under its name and returns it.
    public func evaluate(_ input: String) -> String {
        // Models slip into Python/JS: math.pow, Math.PI, np.sqrt. The
        // namespace prefix is noise to this grammar; the lowercase retries
        // in resolve/apply then land PI on pi and Pow on pow. A trailing
        // "= ?" / "?" / "=" is textbook notation ("1000 * 1.05^22 = ?"),
        // not an assignment -- the 0.8B burned a whole round budget on it.
        let text = input
            .replacingOccurrences(of: #"(?i)\b(math|np)\."#, with: "",
                                  options: .regularExpression)
            .replacingOccurrences(of: #"\s*(=\s*\?+|\?+|=)\s*$"#, with: "",
                                  options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var result = "error: empty expression"
        if !text.isEmpty {
            if let eq = text.firstIndex(of: "=") {
                let name = text[..<eq].trimmingCharacters(in: .whitespaces)
                let rhs = String(text[text.index(after: eq)...])
                let tail = rhs.trimmingCharacters(in: .whitespaces)
                if name == "i" || tail == "i" {
                    result = "error: 'i' is the imaginary unit and cannot "
                        + "be assigned"
                } else if Calculator.isName(name) {
                    result = assign(name, rhs)
                } else if Calculator.isName(tail) {
                    // Reversed assignment ("350 * 0.4 = X1"): models name
                    // the RESULT after the expression.
                    result = assign(tail, name)
                } else {
                    result = "error: '\(name)' is not a valid variable name"
                }
            } else {
                result = evaluated(text)
            }
        }
        return result
    }

    // Store `expr`'s value under `name` (only when finite) and return it.
    private func assign(_ name: String, _ expr: String) -> String {
        let value = Calculator.value(of: expr, vars: context)
        var result = "error: cannot evaluate the expression"
        if let value, value.isFinite {
            context[name] = value
            result = Calculator.format(value)
        } else if value != nil {
            result = "error: result is not a finite number"
        }
        return result
    }

    private func evaluated(_ expr: String) -> String {
        let value = Calculator.value(of: expr, vars: context)
        var result = "error: cannot evaluate the expression"
        if let value {
            result = Calculator.format(value)
        }
        return result
    }

    // ---- pure evaluation core (no actor state) --------------------------

    static func value(of expr: String,
                      vars: [String: Cx]) -> Cx? {
        Parser(tokens: lex(expr), vars: vars).parse()
    }

    // A name is a letter/underscore then letters/digits/underscores.
    static func isName(_ s: String) -> Bool {
        var result = !s.isEmpty
        for (i, c) in s.enumerated() {
            let head = c == "_" || c.isLetter
            result = result && (i == 0 ? head : head || c.isNumber)
        }
        return result
    }

    // Resolve a name: a user variable shadows a built-in constant; a missed
    // constant retries lowercased (PI, E).
    static func resolve(_ name: String, _ vars: [String: Cx]) -> Cx? {
        vars[name] ?? constants[name] ?? constants[name.lowercased()]
    }

    // Apply a built-in function; nil for an unknown name, wrong arity, or a
    // real-only function given a complex argument. Case-folded (Pow, SQRT).
    // log is base-10; log(x, b) is base-b; ln is natural (principal branch
    // on complex / negative input).
    static func apply(_ fn: String, _ a: [Cx]) -> Cx? {
        let name = fn.lowercased()
        var r: Cx? = nil
        switch (name, a.count) {
        case ("sin", 1): r = Cx.sin(a[0])
        case ("cos", 1): r = Cx.cos(a[0])
        case ("tan", 1): r = Cx.tan(a[0])
        case ("sinh", 1): r = Cx.sinh(a[0])
        case ("cosh", 1): r = Cx.cosh(a[0])
        case ("tanh", 1): r = Cx.tanh(a[0])
        case ("exp", 1): r = Cx.exp(a[0])
        case ("ln", 1): r = Cx.ln(a[0])
        case ("log", 1): r = Cx.div(Cx.ln(a[0]), Cx(Foundation.log(10.0)))
        case ("log", 2): r = Cx.div(Cx.ln(a[0]), Cx.ln(a[1]))
        case ("sqrt", 1): r = Cx.sqrt(a[0])
        case ("pow", 2): r = Cx.pow(a[0], a[1])
        case ("abs", 1): r = Cx(Foundation.hypot(a[0].re, a[0].im))
        case ("re", 1): r = Cx(a[0].re)
        case ("im", 1): r = Cx(a[0].im)
        case ("conj", 1): r = Cx(a[0].re, -a[0].im)
        case ("arg", 1): r = Cx(Foundation.atan2(a[0].im + 0, a[0].re))
        default: r = applyReal(name, a)
        }
        return r
    }

    // The real-only tail of the function table: defined for real arguments,
    // nil (-> error) when any argument carries an imaginary part.
    private static func applyReal(_ name: String, _ a: [Cx]) -> Cx? {
        var r: Double? = nil
        if a.allSatisfy({ v in v.im == 0 }) {
            let x = a.map { v in v.re }
            switch (name, x.count) {
            case ("asin", 1): r = asin(x[0])
            case ("acos", 1): r = acos(x[0])
            case ("atan", 1): r = atan(x[0])
            case ("log2", 1): r = log2(x[0])
            case ("log10", 1): r = log10(x[0])
            case ("cbrt", 1): r = cbrt(x[0])
            case ("floor", 1): r = floor(x[0])
            case ("ceil", 1): r = ceil(x[0])
            case ("round", 1): r = x[0].rounded()
            case ("trunc", 1): r = trunc(x[0])
            case ("sign", 1): r = x[0] > 0 ? 1 : (x[0] < 0 ? -1 : 0)
            case ("hypot", 2): r = hypot(x[0], x[1])
            case ("atan2", 2): r = atan2(x[0], x[1])
            case ("mod", 2):
                r = x[1] == 0 ? .nan
                    : x[0].truncatingRemainder(dividingBy: x[1])
            case ("min", 2): r = Swift.min(x[0], x[1])
            case ("max", 2): r = Swift.max(x[0], x[1])
            default: r = nil
            }
        }
        return r.map { v in Cx(v) }
    }

    // Integer-valued results print without a trailing ".0"; NaN / infinity map
    // to an error the model can read rather than the literal "nan" / "inf".
    // A complex value prints "a + bi" with the near-zero cross component
    // snapped away (exp(i*pi) is (-1, 1.2e-16) in floating point -> "-1"),
    // so the principal-branch formulas read as the exact results they mean.
    static func format(_ v: Cx) -> String {
        var result: String
        let z = snapped(v)
        if z.re.isNaN || z.im.isNaN {
            result = "error: undefined result"
        } else if z.re.isInfinite || z.im.isInfinite {
            result = "error: result is infinite"
        } else if z.im == 0 {
            result = formatReal(z.re)
        } else if z.re == 0 {
            result = formatImaginary(z.im)
        } else {
            let sign = z.im < 0 ? " - " : " + "
            result = formatReal(z.re) + sign + formatImaginary(abs(z.im))
        }
        return result
    }

    // Zero a component that is negligible against the other. Purely real
    // values (im == 0 exactly, the every-real-input case) pass through
    // untouched, so real formatting is byte-identical to the real-only
    // calculator's.
    private static func snapped(_ v: Cx) -> Cx {
        var z = v
        if z.im != 0 {
            let scale = Swift.max(abs(z.re), abs(z.im), 1)
            if abs(z.im) <= 1e-12 * scale { z.im = 0 }
            if abs(z.re) <= 1e-12 * scale { z.re = 0 }
        }
        return z
    }

    // 12 significant digits, the classic calculator display: binary float
    // dust (350 * 0.55 -> 192.50000000000003) rounds away instead of being
    // echoed into the model's answer.
    private static func formatReal(_ v: Double) -> String {
        v == v.rounded() && abs(v) < 1e15
            ? String(Int64(v))
            : String(format: "%.12g", v)
    }

    private static func formatImaginary(_ v: Double) -> String {
        var result = formatReal(v) + "i"
        if v == 1 {
            result = "i"
        } else if v == -1 {
            result = "-i"
        }
        return result
    }
}

// A complex value with the principal-branch operations the parser needs.
// Real arithmetic stays exact: both operands im == 0 keeps the result's
// im == 0 (no rounding can leak in), and the trig / exp functions take a
// real fast path, so the calculator's real behavior is unchanged.
struct Cx: Equatable {
    var re: Double
    var im: Double

    init(_ re: Double, _ im: Double = 0) {
        self.re = re
        self.im = im
    }

    var isFinite: Bool { re.isFinite && im.isFinite }

    static func add(_ a: Cx, _ b: Cx) -> Cx {
        Cx(a.re + b.re, a.im + b.im)
    }

    static func sub(_ a: Cx, _ b: Cx) -> Cx {
        Cx(a.re - b.re, a.im - b.im)
    }

    static func mul(_ a: Cx, _ b: Cx) -> Cx {
        Cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    }

    static func div(_ a: Cx, _ b: Cx) -> Cx {
        let den = b.re * b.re + b.im * b.im
        return Cx((a.re * b.re + a.im * b.im) / den,
                  (a.im * b.re - a.re * b.im) / den)
    }

    static func neg(_ a: Cx) -> Cx { Cx(-a.re, -a.im) }

    static func exp(_ z: Cx) -> Cx {
        let r = Foundation.exp(z.re)
        return z.im == 0
            ? Cx(r)
            : Cx(r * Foundation.cos(z.im), r * Foundation.sin(z.im))
    }

    // Principal branch: ln|z| + i*arg(z), so ln(-1) = i*pi and a positive
    // real is exact. The `+ 0` normalizes a negative zero (unary minus on a
    // real produces im == -0.0), which would flip atan2 onto the -pi branch.
    static func ln(_ z: Cx) -> Cx {
        z.im == 0 && z.re > 0
            ? Cx(Foundation.log(z.re))
            : Cx(Foundation.log(Foundation.hypot(z.re, z.im)),
                 Foundation.atan2(z.im + 0, z.re))
    }

    static func sqrt(_ z: Cx) -> Cx {
        let result: Cx
        if z.im == 0 {
            result = z.re >= 0
                ? Cx(Foundation.sqrt(z.re))
                : Cx(0, Foundation.sqrt(-z.re))
        } else {
            let r = Foundation.hypot(z.re, z.im)
            let s = z.im < 0 ? -1.0 : 1.0
            result = Cx(Foundation.sqrt((r + z.re) / 2),
                        s * Foundation.sqrt((r - z.re) / 2))
        }
        return result
    }

    // Real^real keeps the real pow (so 2^10 is exact and 0^0 matches libm);
    // a real result NaN with a negative base -- and any complex operand --
    // takes the principal branch exp(w * ln z). 0^w stays 0 only for a
    // positive real exponent.
    static func pow(_ z: Cx, _ w: Cx) -> Cx {
        let result: Cx
        let realPow = z.im == 0 && w.im == 0
            ? Foundation.pow(z.re, w.re) : Double.nan
        if z.im == 0 && w.im == 0 && (!realPow.isNaN || z.re >= 0) {
            result = Cx(realPow)
        } else if z.re == 0 && z.im == 0 {
            result = w.im == 0 && w.re > 0 ? Cx(0) : Cx(.nan)
        } else {
            result = exp(mul(w, ln(z)))
        }
        return result
    }

    static func sin(_ z: Cx) -> Cx {
        z.im == 0
            ? Cx(Foundation.sin(z.re))
            : Cx(Foundation.sin(z.re) * Foundation.cosh(z.im),
                 Foundation.cos(z.re) * Foundation.sinh(z.im))
    }

    static func cos(_ z: Cx) -> Cx {
        z.im == 0
            ? Cx(Foundation.cos(z.re))
            : Cx(Foundation.cos(z.re) * Foundation.cosh(z.im),
                 -Foundation.sin(z.re) * Foundation.sinh(z.im))
    }

    static func tan(_ z: Cx) -> Cx {
        z.im == 0 ? Cx(Foundation.tan(z.re)) : div(sin(z), cos(z))
    }

    static func sinh(_ z: Cx) -> Cx {
        z.im == 0
            ? Cx(Foundation.sinh(z.re))
            : Cx(Foundation.sinh(z.re) * Foundation.cos(z.im),
                 Foundation.cosh(z.re) * Foundation.sin(z.im))
    }

    static func cosh(_ z: Cx) -> Cx {
        z.im == 0
            ? Cx(Foundation.cosh(z.re))
            : Cx(Foundation.cosh(z.re) * Foundation.cos(z.im),
                 Foundation.sinh(z.re) * Foundation.sin(z.im))
    }

    static func tanh(_ z: Cx) -> Cx {
        z.im == 0 ? Cx(Foundation.tanh(z.re)) : div(sinh(z), cosh(z))
    }
}

// One lexical token. `.bad` marks an unrecognized character, so the lexer never
// fails -- the parser rejects a stream containing it (no control-flow flag).
private enum Tok: Equatable {
    case num(Double)
    case name(String)
    case sym(Character)
    case bad
}

// Numbers (with optional fraction and eNN exponent), identifiers, the operator
// / paren / comma symbols; anything else becomes `.bad`. A bare `e` (no leading
// digit) lexes as the identifier, so the constant still resolves.
private func lex(_ s: String) -> [Tok] {
    var out: [Tok] = []
    let c = Array(s)
    let n = c.count
    var i = 0
    while i < n {
        let ch = c[i]
        if ch == " " || ch == "\t" || ch == "\n" || ch == "\r" {
            i += 1
        } else if ch.isNumber || ch == "." {
            var j = i
            while j < n && c[j].isNumber { j += 1 }
            if j < n && c[j] == "." {
                j += 1
                while j < n && c[j].isNumber { j += 1 }
            }
            if j < n && (c[j] == "e" || c[j] == "E") {
                var k = j + 1
                if k < n && (c[k] == "+" || c[k] == "-") { k += 1 }
                if k < n && c[k].isNumber {
                    j = k + 1
                    while j < n && c[j].isNumber { j += 1 }
                }
            }
            out.append(Double(String(c[i..<j])).map(Tok.num) ?? .bad)
            i = j
        } else if ch.isLetter || ch == "_" {
            var j = i
            while j < n && (c[j].isLetter || c[j].isNumber || c[j] == "_") {
                j += 1
            }
            out.append(.name(String(c[i..<j])))
            i = j
        } else if ch == "*" && i + 1 < n && c[i + 1] == "*" {
            // Python's power operator, which models emit constantly (i**i).
            out.append(.sym("^"))
            i += 2
        } else if "+-*/^(),%".contains(ch) {
            out.append(.sym(ch))
            i += 1
        } else {
            out.append(.bad)
            i += 1
        }
    }
    return out
}

// Recursive-descent over the token stream. Every rule returns Cx?, and a
// nil (a parse error, or a `.bad` token, or an unknown name) propagates upward,
// so there is no error flag -- the absence of a value IS the error.
private final class Parser {
    private let toks: [Tok]
    private let vars: [String: Cx]
    private var pos = 0

    init(tokens: [Tok], vars: [String: Cx]) {
        self.toks = tokens
        self.vars = vars
    }

    // Full parse: nil unless a value was produced AND every token consumed.
    func parse() -> Cx? {
        let value = expr()
        return (value != nil && pos == toks.count) ? value : nil
    }

    private func peek() -> Tok? { pos < toks.count ? toks[pos] : nil }

    // expr := term (('+' | '-') term)*
    private func expr() -> Cx? {
        var result = term()
        while result != nil, let t = peek(),
              t == .sym("+") || t == .sym("-") {
            pos += 1
            let rhs = term()
            if let l = result, let r = rhs {
                result = t == .sym("+") ? Cx.add(l, r) : Cx.sub(l, r)
            } else {
                result = nil
            }
        }
        return result
    }

    // The next term-level operator: '*' or '/' consumed from the stream, or
    // '*' for an implicit multiplication (a name or '(' directly follows a
    // complete factor -- "2i", "3pi", "2sin(1)", the shapes calculators
    // accept and models emit); nil when the term ends.
    private func termOp() -> Character? {
        var result: Character? = nil
        switch peek() {
        case .sym("*"), .sym("/"):
            pos += 1
            if case .sym(let ch) = toks[pos - 1] { result = ch }
        case .name, .sym("("):
            result = "*"
        default:
            result = nil
        }
        return result
    }

    // term := factor (('*' | '/' | juxtaposition) factor)*
    private func term() -> Cx? {
        var result = factor()
        while result != nil, let op = termOp() {
            let rhs = factor()
            if let l = result, let r = rhs {
                result = op == "/" ? Cx.div(l, r) : Cx.mul(l, r)
            } else {
                result = nil
            }
        }
        return result
    }

    // factor := ('-' | '+') factor | power   (unary sign binds below power, so
    // -2^2 is -(2^2))
    private func factor() -> Cx? {
        var result: Cx? = nil
        if let t = peek(), t == .sym("-") || t == .sym("+") {
            pos += 1
            let v = factor()
            if let v { result = t == .sym("-") ? Cx.neg(v) : v }
        } else {
            result = power()
        }
        return result
    }

    // Postfix percent (5% -> 0.05), only when NO operand follows: "5% + x"
    // is a percentage, while "22 % 7" (modulo intent) keeps its old
    // rejection rather than being silently misread as 0.22 * 7.
    private func percentAhead() -> Bool {
        var result = peek() == .sym("%")
        if result, pos + 1 < toks.count {
            switch toks[pos + 1] {
            case .num, .name, .sym("("): result = false
            default: break
            }
        }
        return result
    }

    // power := primary '%'* ('^' factor)?   (right-associative via factor)
    private func power() -> Cx? {
        var result = primary()
        while result != nil, percentAhead() {
            pos += 1
            result = result.map { v in Cx.div(v, Cx(100)) }
        }
        if result != nil, peek() == .sym("^") {
            pos += 1
            let rhs = factor()
            if let l = result, let r = rhs {
                result = Cx.pow(l, r)
            } else {
                result = nil
            }
        }
        return result
    }

    // primary := number | name | name '(' args ')' | '(' expr ')'
    private func primary() -> Cx? {
        var result: Cx? = nil
        switch peek() {
        case .num(let v):
            pos += 1
            result = Cx(v)
        case .sym("("):
            pos += 1
            let inner = expr()
            if inner != nil, peek() == .sym(")") {
                pos += 1
                result = inner
            }
        case .name(let id):
            pos += 1
            if peek() == .sym("(") {
                pos += 1
                if let args = argList() {
                    result = Calculator.apply(id, args)
                }
            } else {
                result = Calculator.resolve(id, vars)
            }
        default:
            result = nil
        }
        return result
    }

    // Comma-separated expressions up to the closing ')', already past the '('.
    // nil on any bad argument or a missing ')'.
    private func argList() -> [Cx]? {
        var result: [Cx]? = nil
        if peek() == .sym(")") {
            pos += 1
            result = []
        } else {
            var list: [Cx] = []
            var value = expr()
            while let v = value {
                list.append(v)
                if peek() == .sym(",") {
                    pos += 1
                    value = expr()
                } else {
                    value = nil
                }
            }
            if !list.isEmpty, peek() == .sym(")") {
                pos += 1
                result = list
            }
        }
        return result
    }
}
