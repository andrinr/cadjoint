/**
 * Python's `format(x, ".9g")`, which is how the Python WGSL backend spells
 * every float literal. The TypeScript emitter has to spell them the same way
 * for the two modules to be the same text.
 *
 * Python rounds the exact binary value half to even; `toExponential` rounds
 * a tie away from zero. A tie needs the exact value to end in a 5 one digit
 * past the ninth, which a 100-digit expansion shows, so ties are resolved on
 * that expansion.
 */
export function formatG9(x: number): string {
  return formatG(x, 9);
}

export function formatG(x: number, precision: number): string {
  if (Number.isNaN(x)) return "nan";
  if (x === Infinity) return "inf";
  if (x === -Infinity) return "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  const sign = x < 0 ? "-" : "";
  const [digits, exponent] = roundedDigits(Math.abs(x), precision);
  if (exponent < -4 || exponent >= precision) {
    let mantissa = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    mantissa = stripZeros(mantissa);
    const e = Math.abs(exponent);
    return `${sign}${mantissa}e${exponent < 0 ? "-" : "+"}${e < 10 ? "0" : ""}${e}`;
  }
  let fixed: string;
  if (exponent >= 0) {
    const whole = digits.slice(0, exponent + 1).padEnd(exponent + 1, "0");
    const fraction = digits.slice(exponent + 1);
    fixed = fraction ? `${whole}.${fraction}` : whole;
  } else {
    fixed = `0.${"0".repeat(-exponent - 1)}${digits}`;
  }
  return sign + stripZeros(fixed);
}

/** `precision` significant digits of a positive `x`, and the decimal exponent of the first. */
function roundedDigits(x: number, precision: number): [string, number] {
  const exact = x.toExponential(99);
  const [mantissa, exp] = exact.split("e");
  let exponent = Number(exp);
  const all = mantissa.replace(".", "");
  let kept = all.slice(0, precision);
  const rest = all.slice(precision);
  const up =
    rest[0] > "5" ||
    (rest[0] === "5" && (/[1-9]/.test(rest.slice(1)) || Number(kept[kept.length - 1]) % 2 === 1));
  if (up) {
    const bumped = (BigInt(kept) + 1n).toString();
    if (bumped.length > precision) {
      kept = bumped.slice(0, precision);
      exponent += 1;
    } else {
      kept = bumped;
    }
  }
  return [kept, exponent];
}

function stripZeros(text: string): string {
  if (!text.includes(".")) return text;
  return text.replace(/0+$/, "").replace(/\.$/, "");
}
