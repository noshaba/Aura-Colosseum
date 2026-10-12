"""Test helper: deterministic ed25519 wallets that sign Aura challenges like Phantom/Solflare do."""
import base64

from nacl.signing import SigningKey

from aura_bounties import b58encode, create_wallet_challenge


class TestWallet:
    def __init__(self, seed: int):
        self.key = SigningKey(bytes([seed]) * 32)
        self.address = b58encode(bytes(self.key.verify_key))

    def sign(self, message: str) -> str:
        return base64.b64encode(self.key.sign(message.encode('utf-8')).signature).decode()


def signed(conn, wallet: TestWallet, action: str, payload: dict) -> dict:
    """payload plus a fresh challenge signed by `wallet` for exactly that action."""
    challenge = create_wallet_challenge(conn, {**payload, 'action': action})
    return {**payload, 'challenge_nonce': challenge['nonce'], 'wallet_signature': wallet.sign(challenge['message'])}
