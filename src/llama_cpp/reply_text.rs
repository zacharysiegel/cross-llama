/// A reply's bytes as tokens produce them; a token can end partway through a UTF-8 character.
pub struct ReplyText {
    bytes: Vec<u8>,
}

impl ReplyText {
    pub fn new() -> Self {
        ReplyText {
            bytes: Vec::new(),
        }
    }

    pub fn push_piece(&mut self, piece: &[u8]) {
        self.bytes.extend_from_slice(piece);
    }

    /// Leaves out a trailing incomplete character, which the next token may complete.
    pub fn text_so_far(&self) -> String {
        let complete_length: usize = length_without_incomplete_trailing_character(&self.bytes);
        String::from_utf8_lossy(&self.bytes[..complete_length]).into_owned()
    }

    pub fn into_text(self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}

fn length_without_incomplete_trailing_character(bytes: &[u8]) -> usize {
    let mut checked_length: usize = 0;

    loop {
        let decoded: Result<&str, std::str::Utf8Error> = std::str::from_utf8(&bytes[checked_length..]);

        let Err(utf8_error) = decoded
        else {
            return bytes.len();
        };

        match utf8_error.error_len() {
            // The input ends partway through a character.
            None => return checked_length + utf8_error.valid_up_to(),
            Some(invalid_sequence_length) => checked_length += utf8_error.valid_up_to() + invalid_sequence_length,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReplyText;

    #[test]
    fn text_so_far_leaves_out_incomplete_trailing_character() {
        let mut reply_text: ReplyText = ReplyText::new();
        reply_text.push_piece("olá".as_bytes());
        reply_text.push_piece(&"ã".as_bytes()[..1]);

        assert_eq!(reply_text.text_so_far(), "olá");
    }

    #[test]
    fn text_so_far_keeps_invalid_bytes_before_later_text() {
        let mut reply_text: ReplyText = ReplyText::new();
        reply_text.push_piece(&[b'a', 0xFF, b'b']);

        assert_eq!(reply_text.text_so_far(), "a\u{FFFD}b");
    }
}
