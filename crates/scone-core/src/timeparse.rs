//! Date references in natural-language queries (E12): "in May 2023",
//! "on 2023/05/20", "2023-05-20". Real memory questions are
//! time-anchored; retrieval that ignores their dates leaves the temporal
//! class on the floor (measured 68% all-evidence, 2026-08-28).

/// An inclusive ISO date window referenced by a query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DateWindow {
    pub start: String,
    pub end: String,
}

const MONTHS: [(&str, u32); 12] = [
    ("january", 1),
    ("february", 2),
    ("march", 3),
    ("april", 4),
    ("may", 5),
    ("june", 6),
    ("july", 7),
    ("august", 8),
    ("september", 9),
    ("october", 10),
    ("november", 11),
    ("december", 12),
];

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        _ => 28,
    }
}

/// Extract explicit date references from a query.
pub fn date_windows(query: &str) -> Vec<DateWindow> {
    let lower = query.to_lowercase();
    let mut windows = Vec::new();

    // Numeric dates: YYYY/MM/DD or YYYY-MM-DD. Scanned over bytes, not
    // string slices: a ten-byte window can land inside a multi-byte
    // character, and slicing a str there panics. A question written in
    // any language but English used to take the process down.
    let bytes = lower.as_bytes();
    let ascii_digits = |b: &[u8]| b.iter().all(u8::is_ascii_digit);
    let as_str = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    let mut i = 0;
    while i + 10 <= bytes.len() {
        let window = &bytes[i..i + 10];
        let sep_ok = matches!(window[4], b'/' | b'-') && window[7] == window[4];
        if sep_ok
            && ascii_digits(&window[..4])
            && ascii_digits(&window[5..7])
            && ascii_digits(&window[8..10])
        {
            let day = format!(
                "{}-{}-{}",
                as_str(&window[..4]),
                as_str(&window[5..7]),
                as_str(&window[8..10])
            );
            windows.push(DateWindow {
                start: format!("{day}T00:00:00Z"),
                end: format!("{day}T23:59:59Z"),
            });
            i += 10;
            continue;
        }
        i += 1;
    }

    // "May 2023" style month references.
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    for pair in tokens.windows(2) {
        if let Some((_, month)) = MONTHS.iter().find(|(name, _)| *name == pair[0])
            && pair[1].len() == 4
            && pair[1].chars().all(|c| c.is_ascii_digit())
            && let Ok(year) = pair[1].parse::<i32>()
        {
            windows.push(DateWindow {
                start: format!("{year:04}-{month:02}-01T00:00:00Z"),
                end: format!(
                    "{year:04}-{month:02}-{:02}T23:59:59Z",
                    days_in_month(year, *month)
                ),
            });
        }
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ten-byte window can land inside a multi-byte character, and
    /// slicing a str there panics. Recall must survive any text a
    /// person can type, in any language.
    #[test]
    fn non_ascii_text_does_not_panic() {
        // The accented vowel straddles the scan window boundary.
        date_windows("qué pasó con la reunión del equipo en la oficina hoy día ó");
        date_windows("日本語のテキストで日付を探す 2023/05/20 まで");
        date_windows("emoji 🧠🥐 memory 2024-02-29 leap");
        date_windows("ó");
        date_windows("");
        // Dates still parse when multi-byte characters surround them.
        let w = date_windows("reunión del 2023-05-20 en Madrid");
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].start, "2023-05-20T00:00:00Z");
    }

    #[test]
    fn parses_numeric_and_month_references() {
        let w = date_windows("what did I say on 2023/05/20 about the trip");
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].start, "2023-05-20T00:00:00Z");
        let w = date_windows("decisions from May 2023 and June 2024?");
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].end, "2023-05-31T23:59:59Z");
        assert_eq!(w[1].end, "2024-06-30T23:59:59Z");
        let w = date_windows("iso style 2024-02-29 leap day");
        assert_eq!(w[0].start, "2024-02-29T00:00:00Z");
    }

    #[test]
    fn plain_queries_have_no_windows() {
        assert!(date_windows("what tools does mark prefer").is_empty());
        assert!(date_windows("may I ask about the 2023 goals").is_empty());
    }
}
