import logging
import time
from typing import Dict, List, Optional

from pylabrobot.plate_reading.agilent.biotek_backend import BioTekPlateReaderBackend
from pylabrobot.resources import Plate, Well

logger = logging.getLogger(__name__)


class SynergyHTBackend(BioTekPlateReaderBackend):
  """Backend for the BioTek Synergy HT plate reader.

  The Synergy HT uses the same FTDI serial-over-USB interface as the H1, but
  reads absorbance via a filter wheel rather than a monochromator. The read
  sequence differs from the H1:

    H1:  D (configure) → O (start) → data until \\x03
    HT:  t (timing)    → S (scan)  → data until \\x03

  Inherited unchanged from BioTekPlateReaderBackend:
    setup(), stop(), get_firmware_version(), get_serial_number(),
    get_current_temperature(), set_plate(), shake(), stop_shaking()

  Overridden from BioTekPlateReaderBackend:
    open()  — base calls _set_slow_mode() before J; HT does not support the
              '&' slow-mode command, so this override skips that call.
    close() — same reason; also preserves the plate-cache reset and optional
              set_plate() call from the base implementation.

  Notes on the init sequence observed in pcap:
    - z <code> commands appear to be diagnostic/status probes; not needed for reads.
    - }, E, n, w, P, o, L, V, {, r, %, k, Q commands during init are not yet
      decoded. They are not required for a basic absorbance read.
    - t + 40000 is the timing/integration setup before S. The 40000 value is
      treated as an opaque parameter for now; it is exposed as integration_param
      so callers can override it if needed.
  """

  @property
  def abs_wavelength_range(self) -> tuple:
    # Filter-based: only the installed filter wavelengths are valid.
    # Query get_available_wavelengths() for the actual list.
    return (400, 700)

  @property
  def supports_heating(self) -> bool:
    return False

  @property
  def supports_cooling(self) -> bool:
    return False

  async def open(self, slow: bool = False):
    """Open the plate carrier. Skips slow-mode setup (HT does not support '&')."""
    return await self.send_command("J")

  async def close(self, plate=None, slow: bool = False):
    """Close the plate carrier. Skips slow-mode setup (HT does not support '&')."""
    self._plate = None
    if plate is not None:
      await self.set_plate(plate)
    return await self.send_command("A")
  
  async def get_current_temperature(self) -> float:
    """Get current temperature in degrees Celsius."""
    resp = await self.send_command("h", timeout=1)
    assert resp is not None
    return int(resp[1:-5])/10

  async def get_available_wavelengths(self) -> List[int]:
    """Query the instrument for its installed absorbance filter wavelengths.

    Sends the 'W' command and parses the comma-separated wavelength list.

    Returns:
      List of available wavelengths in nm (e.g. [600, 595, 525, 510, 410, 562]).
    """
    resp = await self.send_command("W")
    assert resp is not None
    # Response format: \\x06<wl1>,<wl2>,...,0000\\x03
    # Strip ACK and ETX, then split on comma.
    body = resp[1:-1].decode(errors="replace").strip()
    wavelengths = []
    for token in body.split(","):
      token = token.strip()
      try:
        wl = int(token)
        if wl > 0:
          wavelengths.append(wl)
      except ValueError:
        pass
    return wavelengths

  async def read_absorbance(
    self,
    plate: Plate,
    wells: List[Well],
    wavelength: int,
    integration_param: str = "40000",
  ) -> List[Dict]:
    """Read absorbance at a single wavelength for the given wells.

    Protocol sequence (observed in OD600 pcap):
      1. set_plate(plate)      — sends 'y' command with plate geometry
      2. 't' + integration_param — sets up timing/integration
      3. 'S'                   — starts scan; returns timestamp + 96 values

    Args:
      plate: The Plate resource currently on the instrument.
      wells: Wells to read (currently all 96 values are always returned).
      wavelength: Target wavelength in nm; must match an installed filter.
      integration_param: Opaque parameter sent with 't' command (default "40000").

    Returns:
      List with one dict containing:
        wavelength  — int, nm
        data        — List[List[Optional[float]]], [row][col] indexed
        temperature — float, °C
        time        — float, Unix timestamp
    """
    available = await self.get_available_wavelengths()
    if available and wavelength not in available:
      raise ValueError(
        f"SynergyHTBackend: wavelength {wavelength} nm not available. "
        f"Installed filters: {available}"
      )

    await self.set_plate(plate)

    # Step 1: timing setup — send 't', receive ACK, send param, receive response
    await self.send_command("t", integration_param)

    # Step 2: start scan — send 'S', read until ETX
    resp = await self.send_command("S", timeout=60 * 3)
    assert resp is not None

    data = self._parse_scan_response(resp, plate)

    try:
      temp = await self.get_current_temperature()
    except TimeoutError:
      temp = float("nan")

    return [
      {
        "wavelength": wavelength,
        "data": data,
        "temperature": temp,
        "time": time.time(),
      }
    ]

  def _parse_scan_response(
    self, resp: bytes, plate: Plate
  ) -> List[List[Optional[float]]]:
    """Parse the raw 'S' command response into a 2-D well grid.

    Response format (from pcap):
      \\x06<timestamp>,+NNNN,+NNNN,...,+NNNN\\x03

    The timestamp token (e.g. "065:30:00.2") is skipped.  Remaining tokens
    are +NNNN absorbance values in row-major order (A1, A2, ..., H12 for a
    96-well plate).

    Args:
      resp:  Raw bytes from send_command("S").
      plate: Plate resource used to determine grid dimensions.

    Returns:
      2-D list indexed [row][col]; None where no value was received.
    """
    # Strip leading ACK (\\x06) and trailing ETX (\\x03)
    body = resp[1:-1].decode(errors="replace")

    tokens = body.split(",")
    # First token is the timestamp; skip it
    value_tokens = tokens[1:]

    num_rows = plate.num_items_y
    num_cols = plate.num_items_x

    result: List[List[Optional[float]]] = [
      [None for _ in range(num_cols)] for _ in range(num_rows)
    ]

    idx = 0
    for r in range(num_rows):
      for c in range(num_cols):
        if idx >= len(value_tokens):
          break
        token = value_tokens[idx].strip().lstrip("+")
        idx += 1
        try:
          result[r][c] = float(token)
        except ValueError:
          result[r][c] = None

    return result