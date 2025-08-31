export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Clear the authentication cookie
    res.setHeader('Set-Cookie', [
      'token=; HttpOnly; Path=/; Max-Age=0; SameSite=Strict',
      'user=; HttpOnly; Path=/; Max-Age=0; SameSite=Strict'
    ]);

    res.status(200).json({ message: 'Logged out successfully' });
  } catch (error) {
    console.error('Logout error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}