import unittest
from dataclasses import replace

from .game_record.game_record import GameRecord
from .minizero_record import RecordParseError, parse_record
from .move.move import Move

ROOT = "(;GM[go_9x9]RE[-1.000000]OBS[]SZ[9]KM[7.000000]"
GAME = (
    ROOT + "EV[go_9x9_az.pt]DLEN[0-3]"
    ";B[0]P[0:18,81:2]V[-0.2]R[0]"
    ";W[80]P[80:20]V[-0.3]R[0]"
    ";B[81]P[81:17,4:3]V[-0.5]R[0]"
    ";W[81]P[81:20]V[-0.8]R[0])"
)
SELFPLAY = f"SelfPlay true 4 4 -1 {GAME} #"


class MiniZeroRecordTest(unittest.TestCase):
    def test_parses_go_selfplay_message(self) -> None:
        self.assertEqual(
            parse_record(SELFPLAY),
            GameRecord(
                board_size=9,
                komi=7.0,
                black_result=-1,
                moves=(
                    Move(1, 0, ((0, 18.0), (81, 2.0)), -0.2, 0.0),
                    Move(2, 80, ((80, 20.0),), -0.3, 0.0),
                    Move(1, 81, ((81, 17.0), (4, 3.0)), -0.5, 0.0),
                    Move(2, 81, ((81, 20.0),), -0.8, 0.0),
                ),
                sample_range=range(4),
                model_name="go_9x9_az.pt",
                is_complete=True,
            ),
        )

    def test_parses_game_body_with_optional_marker(self) -> None:
        record = parse_record(SELFPLAY)
        self.assertEqual(parse_record(f" {GAME} #\r\n"), record)
        self.assertEqual(parse_record(GAME), replace(record, is_complete=None))

    def test_preserves_partial_record_and_training_range(self) -> None:
        body = ROOT + "DLEN[1-1];B[0];W[80]P[80:0.75,81:0.25];B[81]P[])"
        record = parse_record(f"SelfPlay false 1 3 -1 {body} #")
        self.assertIs(record.is_complete, False)
        self.assertEqual(record.sample_range, range(1, 2))
        self.assertEqual(
            record.moves,
            (Move(1, 0), Move(2, 80, ((80, 0.75), (81, 0.25))), Move(1, 81)),
        )

    def test_handles_escaping_and_whitespace(self) -> None:
        body = ROOT + "\n " + r"EV[model \(v1\)\[go\]\\part\].pt #] ; B[0] C[note;)] )"
        record = parse_record(body)
        self.assertEqual(record.model_name, r"model (v1)[go]\part].pt #")
        self.assertEqual(record.moves, (Move(1, 0),))
        self.assertEqual(record.sample_range, range(1))

    def test_derives_pass_from_board_size(self) -> None:
        record = parse_record("(;GM[go_13x13]SZ[13]KM[6.5]RE[1];B[169]P[169:5])")
        self.assertEqual(record.board_size, 13)
        self.assertEqual(record.komi, 6.5)
        self.assertEqual(record.black_result, 1)
        self.assertEqual(record.moves, (Move(1, 169, ((169, 5.0),)),))

    def test_allows_empty_resignation_record(self) -> None:
        record = parse_record(f"SelfPlay true 0 0 -1 {ROOT}DLEN[0--1]) #")
        self.assertEqual(record.moves, ())
        self.assertEqual(record.sample_range, range(0))
        self.assertEqual(record.black_result, -1)
        self.assertIs(record.is_complete, True)

    def test_rejects_bad_selfplay_header(self) -> None:
        cases = {
            SELFPLAY[:-1]: "Expected 'SelfPlay",
            f"SelfPlay maybe 4 4 -1 {GAME} #": "Expected 'SelfPlay",
            f"SelfPlay true 4 5 -1 {GAME} #": "game length",
            f"SelfPlay true 3 4 -1 {GAME} #": "data length",
            f"SelfPlay true 4 4 +1 {GAME} #": "result disagrees",
        }
        for text, error in cases.items():
            with self.subTest(text=text):
                with self.assertRaisesRegex(RecordParseError, error):
                    parse_record(text)

    def test_rejects_malformed_game_syntax(self) -> None:
        cases = {
            ROOT + ";B[0": "delimiter",
            ROOT + ";B[0]": "delimiter",
            ROOT + ";B[0](;W[1]))": "delimiter",
            GAME + GAME: "Unexpected text",
            ROOT + "SZ[9])": "Duplicate SZ",
            ROOT + "AB[0])": "Setup",
            ROOT + ";B[0]W[1])": "exactly one B or W",
        }
        for text, error in cases.items():
            with self.subTest(text=text):
                with self.assertRaisesRegex(RecordParseError, error):
                    parse_record(text)

    def test_rejects_invalid_game_fields(self) -> None:
        cases = {
            "(;GM[go_9x9]SZ[9]KM[7];B[0])": "Missing RE",
            "(;GM[go_9x9]SZ[9]KM[7]RE[0.5];B[0])": "Go RE",
            "(;GM[go_9x9]SZ[13]KM[7]RE[-1];B[0])": r"Expected GM\[go_13x13\]",
            "(;GM[1]SZ[9]KM[7]RE[-1];B[aa])": "Expected GM",
            ROOT + "DLEN[0-1];B[0])": "DLEN must select",
        }
        for text, error in cases.items():
            with self.subTest(text=text):
                with self.assertRaisesRegex(RecordParseError, error):
                    parse_record(text)

    def test_rejects_invalid_move_fields(self) -> None:
        cases = {
            ";B[82])": r"Move 1.*ID range",
            ";B[-1])": r"Move 1.*nonnegative integer",
            ";B[0]P[82:2])": r"Move 1.*P action ID",
            ";B[0]P[0:1,0:3])": "duplicate action ID",
            ";B[0]P[0:1,])": "requires comma-separated",
            ";B[0]P[0:-1])": "weights must be nonnegative",
            ";B[0]P[0:0])": "positive finite sum",
            ";B[0]P[0:1e309])": "finite number",
            ";B[0]V[nan])": "finite number",
        }
        for fragment, error in cases.items():
            with self.subTest(fragment=fragment):
                with self.assertRaisesRegex(RecordParseError, error):
                    parse_record(ROOT + fragment)


if __name__ == "__main__":
    unittest.main()
