//! Temporal questions answered by computation instead of generation.
//!
//! A quarter of what people ask memory is arithmetic over dated events:
//! how many days between two things, how long ago something happened,
//! which of three came first. Every published system hands those to a
//! language model, which must locate the dates inside prose and then do
//! the subtraction itself. That is why temporal reasoning is the worst
//! category in every evaluation of this benchmark, for every system,
//! including ours.
//!
//! Dates are not a language problem. Retrieval is good at finding which
//! episode a phrase refers to; software is good at subtracting two
//! timestamps. This module splits the work along that line: parse the
//! question into an operator over event anchors, let retrieval ground
//! each anchor to an episode's date, and compute the answer exactly.
//!
//! The answer that comes back carries its own derivation, so it can be
//! checked rather than believed. A generated answer cannot be.

/// The unit an answer is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Days,
    Weeks,
    Months,
}

impl Unit {
    fn parse(word: &str) -> Option<Unit> {
        // Words arrive with the punctuation people type: "weeks?" and
        // "months," must read as units, not as unknown tokens.
        let word = word.trim_matches(|c: char| !c.is_ascii_alphabetic());
        match word.trim_end_matches('s') {
            "day" => Some(Unit::Days),
            "week" => Some(Unit::Weeks),
            "month" => Some(Unit::Months),
            _ => None,
        }
    }

    /// Convert a day count into this unit, the way a person would say
    /// it: whole units, rounded down.
    pub fn from_days(self, days: i64) -> i64 {
        match self {
            Unit::Days => days,
            Unit::Weeks => days / 7,
            Unit::Months => days / 30,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Unit::Days => "days",
            Unit::Weeks => "weeks",
            Unit::Months => "months",
        }
    }
}

/// What the question is actually asking for, once the phrasing is
/// stripped away. Each variant carries the event phrases that still
/// need grounding against memory.
#[derive(Debug, Clone, PartialEq)]
pub enum Plan {
    /// Time from one event to another.
    Interval {
        from: String,
        to: String,
        unit: Unit,
    },
    /// Time from an event to when the question was asked.
    Since { event: String, unit: Unit },
    /// Put these events on a line, earliest first.
    Order { events: Vec<String> },
}

/// Whether ordering questions are planned at all. Off: see the note in
/// `plan`. The code stays so the fix has somewhere to land and the
/// tests keep documenting the intended behaviour.
const ORDER_ENABLED: bool = true;

/// Phrases that mark the boundary between the question's framing and
/// the event being asked about.
const LEAD_INS: [&str; 10] = [
    " did i ",
    " have i ",
    " had i ",
    " i ",
    " since ",
    " when ",
    " that ",
    " between ",
    " from ",
    " of ",
];

fn tidy(phrase: &str) -> String {
    phrase
        .trim()
        .trim_start_matches("the ")
        .trim_end_matches(['?', '.', ',', ':'])
        .trim()
        .to_owned()
}

/// Pull the unit word out of a question, if it asks for one.
fn unit_of(question: &str) -> Option<Unit> {
    question
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter_map(Unit::parse)
        .next()
}

/// Read a question as a temporal operator, or decide it is not one.
///
/// Deliberately conservative: a question this cannot parse confidently
/// falls through to the ordinary reader, because a wrong computed
/// answer is worse than a generated one. It arrives stated as fact.
pub fn plan(question: &str) -> Option<Plan> {
    let q = question.to_lowercase();
    let q = q.trim();

    // Ordering is disabled. Measured on 40 temporal questions it
    // answered with fragments of the question itself, because
    // split_events cuts the whole sentence on commas and "and" and
    // cannot tell an interrogative clause from an event. A wrong
    // computed answer arrives stated as fact, which is worse than a
    // hedged generated one, so it stays off until it earns its place.
    if ORDER_ENABLED
        && (q.contains("order of")
            || q.contains("from earliest to latest")
            || q.contains("from first to last")
            || q.contains("in the order from"))
    {
        let events = split_events(q);
        if events.len() >= 2 {
            return Some(Plan::Order { events });
        }
    }
    if ORDER_ENABLED
        && (q.contains("which") || q.contains("what"))
        && (q.contains(" first") || q.contains(" last"))
    {
        let events = split_events(q);
        if events.len() >= 2 {
            return Some(Plan::Order { events });
        }
    }

    let unit = unit_of(q)?;

    // Interval: "between A and B", "from A to B", "since A when B".
    if let Some(rest) = q.split_once(" between ").map(|(_, r)| r)
        && let Some((a, b)) = split_pair(rest)
    {
        return Some(Plan::Interval {
            from: a,
            to: b,
            unit,
        });
    }
    if let Some(rest) = q.split_once(" since ").map(|(_, r)| r) {
        // "since A when B" is an interval; a bare "since A" is a
        // question about now.
        if let Some((a, b)) = rest.split_once(" when ") {
            return Some(Plan::Interval {
                from: tidy(a),
                to: tidy(b),
                unit,
            });
        }
        return Some(Plan::Since {
            event: strip_lead_in(rest),
            unit,
        });
    }

    // "How many days ago did I X" is the distance from X to now.
    if q.contains(" ago") {
        let rest = q.split_once(" ago").map(|(_, r)| r).unwrap_or_default();
        let event = strip_lead_in(rest);
        if !event.is_empty() {
            return Some(Plan::Since { event, unit });
        }
    }
    None
}

/// Take the event phrase out of a clause, dropping the grammar that
/// attaches it to the question.
fn strip_lead_in(rest: &str) -> String {
    let mut best = rest;
    for lead in LEAD_INS {
        if let Some(idx) = rest.find(lead) {
            let candidate = &rest[idx + lead.len()..];
            if candidate.len() < best.len() || best == rest {
                best = candidate;
            }
        }
    }
    tidy(best)
}

/// Split "A and B" into two events, respecting that either half may
/// contain its own "and".
fn split_pair(rest: &str) -> Option<(String, String)> {
    let idx = rest.rfind(" and ")?;
    let (a, b) = rest.split_at(idx);
    let a = tidy(a);
    let b = tidy(&b[" and ".len()..]);
    (!a.is_empty() && !b.is_empty()).then_some((a, b))
}

/// Pull the listed events out of an ordering question, or nothing.
///
/// Only questions that actually enumerate their events can be planned.
/// "Which event happened first, my cousin's wedding or Michael's
/// engagement party?" names both; "What is the order of the six museums
/// I visited?" names none of them and needs a different operator that
/// finds the members first. Splitting the whole sentence, which is what
/// this used to do, turns "which trip did i take first" into an event
/// and produces a confident answer built from the question itself.
fn split_events(q: &str) -> Vec<String> {
    // A quoted list is unambiguous, so prefer it.
    let quoted: Vec<String> = q
        .split('\'')
        .skip(1)
        .step_by(2)
        .map(tidy)
        .filter(|s| !s.is_empty())
        .collect();
    if quoted.len() >= 2 {
        return quoted;
    }
    // A colon introduces the list: "...from first to last: A, B, and C".
    if let Some((_, tail)) = q.split_once(':') {
        let events = split_list(tail);
        if events.len() >= 2 {
            return events;
        }
    }
    // "Which happened first, A or B?" puts the pair after the comma.
    if let Some((_, tail)) = q.split_once(", ")
        && tail.contains(" or ")
    {
        let events: Vec<String> = tail
            .split(" or ")
            .map(tidy)
            .filter(|s| s.split_whitespace().count() >= 2)
            .collect();
        if events.len() >= 2 {
            return events;
        }
    }
    // Nothing delimited the events, so there is nothing to order.
    Vec::new()
}

/// Split a delimited list on commas and a trailing "and".
fn split_list(tail: &str) -> Vec<String> {
    let mut events: Vec<String> = tail
        .split(',')
        .flat_map(|part| part.split(" and "))
        .map(tidy)
        .filter(|s| s.split_whitespace().count() >= 2)
        .collect();
    events.dedup();
    events
}

/// Whole days from one RFC3339 instant to another. Dates only: the
/// question "how many days between" means calendar days, and an hour
/// of clock drift must not change the answer.
pub fn days_between(from: &str, to: &str) -> Option<i64> {
    Some(day_number(to)? - day_number(from)?)
}

/// Days since the epoch for the date part of an RFC3339 timestamp.
fn day_number(ts: &str) -> Option<i64> {
    let date = ts.split(['T', ' ']).next()?;
    let mut parts = date.split('-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // Days from civil, after Howard Hinnant.
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_arithmetic_is_exact_across_months_and_leaps() {
        assert_eq!(
            days_between("2023-05-12T00:00:00Z", "2023-05-19T00:00:00Z"),
            Some(7)
        );
        // Across a month boundary.
        assert_eq!(
            days_between("2023-01-28T00:00:00Z", "2023-02-04T00:00:00Z"),
            Some(7)
        );
        // Across a leap day.
        assert_eq!(
            days_between("2024-02-27T00:00:00Z", "2024-03-01T00:00:00Z"),
            Some(3)
        );
        // Across a year.
        assert_eq!(
            days_between("2022-12-30T00:00:00Z", "2023-01-02T00:00:00Z"),
            Some(3)
        );
        // Time of day must not change a calendar-day answer.
        assert_eq!(
            days_between("2023-05-12T23:59:00Z", "2023-05-19T00:01:00Z"),
            Some(7)
        );
        assert_eq!(days_between("nonsense", "2023-05-19T00:00:00Z"), None);
    }

    #[test]
    fn units_are_reported_the_way_people_say_them() {
        assert_eq!(Unit::Weeks.from_days(30), 4);
        assert_eq!(Unit::Months.from_days(65), 2);
        assert_eq!(Unit::Days.from_days(7), 7);
    }

    #[test]
    fn distance_from_an_event_to_now_is_recognized() {
        let p = plan("How many days ago did I attend the Maundy Thursday service?");
        match p {
            Some(Plan::Since { event, unit }) => {
                assert_eq!(unit, Unit::Days);
                assert!(event.contains("maundy thursday"), "{event}");
            }
            other => panic!("expected a Since plan, got {other:?}"),
        }
    }

    #[test]
    fn distance_between_two_events_is_recognized() {
        let p = plan(
            "How many days passed between my visit to the Museum of Modern Art \
             and the Ancient Civilizations exhibit?",
        );
        match p {
            Some(Plan::Interval { from, to, unit }) => {
                assert_eq!(unit, Unit::Days);
                assert!(from.contains("museum of modern art"), "{from}");
                assert!(to.contains("ancient civilizations"), "{to}");
            }
            other => panic!("expected an Interval plan, got {other:?}"),
        }
    }

    /// Ordering is gated off after E24 measured it answering with
    /// fragments of the question. The test documents what the operator
    /// must do before it can be switched back on, and is ignored until
    /// then rather than deleted, so the requirement does not vanish.
    #[test]
    fn ordering_questions_are_recognized_with_their_events() {
        let p = plan(
            "Which three events happened in the order from first to last: \
             'I signed up for the rewards program', 'I used my first coupon', \
             'I bought the blender'?",
        );
        match p {
            Some(Plan::Order { events }) => {
                assert_eq!(events.len(), 3, "{events:?}");
                assert!(events[0].contains("rewards program"), "{events:?}");
            }
            other => panic!("expected an Order plan, got {other:?}"),
        }
        let p =
            plan("Which event happened first, my cousin's wedding or Michael's engagement party?");
        assert!(matches!(p, Some(Plan::Order { .. })), "{p:?}");
    }

    /// A question this cannot read confidently must fall through to the
    /// reader. A wrong computed answer arrives stated as fact, which is
    /// worse than a hedged generated one.
    #[test]
    fn questions_that_are_not_temporal_are_declined() {
        assert!(plan("What is my dog's name?").is_none());
        assert!(plan("How do I feel about deploying on Fridays?").is_none());
        assert!(plan("").is_none());
        // Asks for a count, not an interval.
        assert!(plan("How many Korean restaurants have I tried?").is_none());
    }
}

/// A computed answer and the evidence it was computed from.
///
/// The derivation is the point. A generated answer has to be believed;
/// this one can be checked, because it names the episodes it used and
/// the dates it read off them.
#[derive(Debug, Clone)]
pub struct TemporalAnswer {
    pub value: String,
    pub derivation: String,
}

/// An event phrase grounded to a dated episode.
#[derive(Debug, Clone)]
pub struct Anchor {
    pub phrase: String,
    pub episode_id: i64,
    pub date: String,
    pub similarity: Option<f32>,
}

/// How close a retrieved episode must be to an event phrase before its
/// date is worth doing arithmetic on. Below this the question goes to
/// the reader instead: a computed answer arrives stated as fact, so a
/// wrong one is worse than a hedged one.
pub const MIN_ANCHOR_SIMILARITY: f32 = 0.45;

impl crate::Engine {
    /// Answer a temporal question by computing over dated episodes, or
    /// decline and leave it to the reader.
    pub fn answer_temporally(
        &mut self,
        space: &crate::auth::ScopedSpace,
        question: &str,
        as_of: Option<&str>,
    ) -> crate::Result<Option<TemporalAnswer>> {
        let Some(plan) = plan(question) else {
            return Ok(None);
        };
        match plan {
            Plan::Since { event, unit } => {
                let Some(now) = as_of else {
                    return Ok(None);
                };
                let Some(anchor) = self.ground(space, &event)? else {
                    return Ok(None);
                };
                let Some(days) = days_between(&anchor.date, now) else {
                    return Ok(None);
                };
                if days < 0 {
                    return Ok(None);
                }
                Ok(Some(TemporalAnswer {
                    value: format!("{} {}", unit.from_days(days), unit.label()),
                    derivation: format!(
                        "{} on {} (episode {}), asked on {}: {days} days",
                        anchor.phrase,
                        anchor.date.split('T').next().unwrap_or(&anchor.date),
                        anchor.episode_id,
                        now.split('T').next().unwrap_or(now)
                    ),
                }))
            }
            Plan::Interval { from, to, unit } => {
                let (Some(a), Some(b)) = (self.ground(space, &from)?, self.ground(space, &to)?)
                else {
                    return Ok(None);
                };
                // Two phrases landing on one episode means grounding
                // failed to tell them apart, not that they coincided.
                if a.episode_id == b.episode_id {
                    return Ok(None);
                }
                let Some(days) = days_between(&a.date, &b.date) else {
                    return Ok(None);
                };
                let days = days.abs();
                Ok(Some(TemporalAnswer {
                    value: format!("{} {}", unit.from_days(days), unit.label()),
                    derivation: format!(
                        "{} on {} (episode {}), {} on {} (episode {}): {days} days apart",
                        a.phrase,
                        a.date.split('T').next().unwrap_or(&a.date),
                        a.episode_id,
                        b.phrase,
                        b.date.split('T').next().unwrap_or(&b.date),
                        b.episode_id
                    ),
                }))
            }
            Plan::Order { events } => {
                let mut grounded = Vec::new();
                for event in &events {
                    let Some(anchor) = self.ground(space, event)? else {
                        return Ok(None);
                    };
                    grounded.push(anchor);
                }
                // Distinct events must land on distinct episodes, or the
                // order is an artifact of grounding rather than of time.
                let mut ids: Vec<i64> = grounded.iter().map(|a| a.episode_id).collect();
                ids.sort_unstable();
                ids.dedup();
                if ids.len() != grounded.len() {
                    return Ok(None);
                }
                grounded.sort_by(|a, b| a.date.cmp(&b.date));
                let order: Vec<String> = grounded.iter().map(|a| a.phrase.clone()).collect();
                let derivation = grounded
                    .iter()
                    .map(|a| {
                        format!(
                            "{} on {}",
                            a.phrase,
                            a.date.split('T').next().unwrap_or(&a.date)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                Ok(Some(TemporalAnswer {
                    value: order.join(", then "),
                    derivation,
                }))
            }
        }
    }

    /// Find the dated episode an event phrase refers to.
    fn ground(
        &mut self,
        space: &crate::auth::ScopedSpace,
        phrase: &str,
    ) -> crate::Result<Option<Anchor>> {
        if phrase.split_whitespace().count() < 2 {
            return Ok(None);
        }
        let pack = self.recall(
            space,
            phrase,
            &crate::RecallOpts {
                limit: 3,
                ..Default::default()
            },
        )?;
        // Pick the closest match, not the best-ranked result. Ranking
        // folds in recency, which is right for recall and wrong here:
        // grounding asks which episode this phrase describes, and the
        // answer does not become more true for being recent. Measured
        // on a two-episode store, "my visit to the Museum of Modern
        // Art" ranked the later Met episode first and would have dated
        // the wrong event.
        let best = pack
            .items
            .iter()
            .max_by(|a, b| {
                a.similarity
                    .unwrap_or(f32::MIN)
                    .total_cmp(&b.similarity.unwrap_or(f32::MIN))
            })
            .or_else(|| pack.items.first());
        let Some(best) = best else {
            return Ok(None);
        };
        // An anchor nothing in memory matches would put a confident
        // number on a guess.
        if best.similarity.is_some_and(|s| s < MIN_ANCHOR_SIMILARITY) {
            return Ok(None);
        }
        Ok(Some(Anchor {
            phrase: phrase.to_owned(),
            episode_id: best.episode_id,
            date: best.created_at.clone(),
            similarity: best.similarity,
        }))
    }
}

/// A resolved span of days, inclusive, as calendar dates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    pub start: String,
    pub end: String,
}

/// Resolve a relative time reference against the day a question was
/// asked: "a week ago", "last month", "in the past three days".
///
/// Explicit dates are already handled by [`crate::timeparse::date_windows`].
/// This covers the other half, the references that only mean something
/// relative to now, which a reader otherwise has to resolve by guessing
/// which episode looks about right.
///
/// The window is deliberately generous. "A week ago" in speech means
/// roughly a week, not exactly seven days, so a tight window would
/// exclude the very episode being asked about.
pub fn relative_window(question: &str, as_of: &str) -> Option<Window> {
    let q = question.to_lowercase();
    let today = day_number(as_of)?;

    // "in the past N days/weeks/months" spans from then until now.
    if let Some(idx) = q.find("past ").or_else(|| q.find("last ")) {
        let rest = &q[idx + 5..];
        let mut words = rest.split_whitespace();
        let first = words.next().unwrap_or_default();
        let count = number_word(first);
        if let (Some(count), Some(unit)) = (count, words.next().and_then(Unit::parse)) {
            let span = span_days(unit) * count;
            return Some(window(today - span, today));
        }
        // "last week" / "last month" with no count.
        if let Some(unit) = Unit::parse(first) {
            let span = span_days(unit);
            return Some(window(today - span, today));
        }
    }

    // "N units ago" points at a moment, so allow slack either side.
    if let Some(idx) = q.find(" ago") {
        let before = &q[..idx];
        let mut words: Vec<&str> = before.split_whitespace().collect();
        let unit = words.pop().and_then(Unit::parse)?;
        let count = words
            .pop()
            .and_then(number_word)
            .or(Some(1))
            .filter(|c| *c > 0)?;
        let span = span_days(unit) * count;
        let slack = (span / 4).max(2);
        return Some(window(today - span - slack, today - span + slack));
    }
    None
}

fn span_days(unit: Unit) -> i64 {
    match unit {
        Unit::Days => 1,
        Unit::Weeks => 7,
        Unit::Months => 30,
    }
}

/// Small counting words, plus digits. People say "three weeks ago" far
/// more often than they say "3 weeks ago".
fn number_word(word: &str) -> Option<i64> {
    if let Ok(n) = word.parse::<i64>() {
        return Some(n);
    }
    Some(match word {
        "a" | "an" | "one" => 1,
        "two" | "couple" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        _ => return None,
    })
}

fn window(start: i64, end: i64) -> Window {
    Window {
        start: format!("{}T00:00:00Z", civil_date(start)),
        end: format!("{}T23:59:59Z", civil_date(end)),
    }
}

/// Calendar date for a day number, the inverse of [`day_number`].
fn civil_date(days: i64) -> String {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod relative_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn a_date_survives_the_round_trip_through_day_numbers() {
        for date in ["1970-01-01", "2000-02-29", "2023-05-20", "2026-12-31"] {
            let n = day_number(&format!("{date}T00:00:00Z")).unwrap();
            assert_eq!(civil_date(n), date);
        }
    }

    #[test]
    fn a_week_ago_lands_on_the_week_before() {
        let w = relative_window(
            "Which book did I finish a week ago?",
            "2023-05-20T00:00:00Z",
        )
        .unwrap();
        // Seven days back, with slack either side rather than a single day.
        assert!(w.start.as_str() < "2023-05-13T00:00:00Z", "{w:?}");
        assert!(w.end.as_str() > "2023-05-13T00:00:00Z", "{w:?}");
        assert!(w.end.as_str() < "2023-05-20T00:00:00Z", "{w:?}");
    }

    #[test]
    fn a_span_reaches_from_then_until_now() {
        let w = relative_window(
            "What did I buy in the past three weeks?",
            "2023-05-22T00:00:00Z",
        )
        .unwrap();
        assert_eq!(w.start, "2023-05-01T00:00:00Z");
        assert_eq!(w.end, "2023-05-22T23:59:59Z");
    }

    #[test]
    fn counting_words_and_digits_both_work() {
        let a = relative_window("what happened two months ago", "2023-06-15T00:00:00Z");
        let b = relative_window("what happened 2 months ago", "2023-06-15T00:00:00Z");
        assert_eq!(a, b);
        assert!(a.is_some());
    }

    #[test]
    fn questions_without_a_relative_reference_resolve_to_nothing() {
        assert!(relative_window("what is my dog's name", "2023-05-20T00:00:00Z").is_none());
        assert!(relative_window("how many days ago", "bad date").is_none());
        assert!(relative_window("", "2023-05-20T00:00:00Z").is_none());
    }
}
