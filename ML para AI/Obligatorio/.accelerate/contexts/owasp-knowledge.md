# List of OWASP Top 10 Vulnerabilities

1. A01:2021-Broken Access Control
Description: Access control enforces policy such that users cannot act outside of their intended permissions.

- Examples:
- Missing authorization checks
- Insecure direct object references (IDOR)
- Path traversal vulnerabilities
- Elevation of privilege flaws
- Bypassing access control checks
- Permission manipulation
- Relying solely on client-side authorization
- Unprotected API endpoints
- Missing UI gating for sensitive actions


2. A02:2021-Cryptographic Failures

Description: Failures related to cryptography that often lead to sensitive data exposure or system compromise.

- Examples:
- Weak encryption algorithms
- Hardcoded secrets or keys
- Missing encryption for sensitive data
- Inadequate key management
- Insufficient entropy
- Cleartext transmission of sensitive data
- Use of insecure hash functions (MD5/SHA1)
- Missing TLS enforcement


3. A03:2021-Injection

Description: Injection flaws like SQL, NoSQL, OS, and LDAP injection occur when untrusted data is sent to an interpreter.

- Examples:
- SQL injection
- NoSQL injection
- Command injection
- Cross-site scripting (XSS)
- Template injection
- LDAP injection
- GraphQL injection
- HTML injection (e.g., dangerouslySetInnerHTML)
- CSS/URL injection
- DOM-based XSS (e.g., innerHTML, insertAdjacentHTML)


4. A04:2021-Insecure Design

Description: Flaws related to design and architectural security failures.

- Examples:
- Missing business logic validation
- Lack of rate limiting
- Insecure business flows
- Insufficient threat modeling
- Lack of security controls
- Single layer defenses
- Open redirect flows based on unsanitized input
- Missing CSRF protections
- Lack of multi-step validation flows
- UI bypass weaknesses


5. A05:2021-Security Misconfiguration

Description: Security misconfiguration is the most commonly seen issue, often a result of insecure default settings.

- Examples:
- Default credentials
- Open cloud storage
- Verbose error messages
- Missing security headers
- Unnecessary features enabled
- Outdated software
- Missing CORS restrictions
- Debug modes enabled in production
- Fetching or embedding HTTP resources on an HTTPS page (Mixed content)
- Using postMessage with wildcard origin '*'
- Setting document.cookie without HttpOnly, Secure, or SameSite flags
- Access-Control-Allow-Origin: * misconfiguration
- Missing Content Security Policy (CSP)


6. A06:2021-Vulnerable and Outdated Components

Description: Using components with known vulnerabilities may undermine application defenses.

- Examples:
- Outdated libraries and frameworks
- Vulnerable dependencies
- Unmaintained libraries
- Outdated client-side libraries
- Lack of patch management
- Deprecated OS packages
- Unverified third-party frontend scripts
- Outdated Docker base images


7. A07:2021-Identification and Authentication Failures

Description: Confirmation of the user's identity, authentication, and session management.

- Examples:
- Weak password policies
- Credential stuffing vulnerabilities
- Brute force susceptibility
- Session fixation
- Missing MFA
- Weak session management
- Tokens stored in localStorage without HttpOnly/Secure flags
- Storing tokens in sessionStorage without appropriate security attributes
- No account lockout after authentication failures


8. A08:2021-Software and Data Integrity Failures

Description: Software and data integrity failures relate to code and infrastructure that doesn't protect against integrity violations.

- Examples:
- Insecure deserialization
- Unsigned code execution
- Auto-update vulnerabilities
- CI/CD pipeline weaknesses
- Tamperable data
- Unsafe JSON.parse of untrusted input
- Dynamic module loading without integrity checks


9. A09:2021-Security Logging and Monitoring Failures

Description: This category helps detect, escalate, and respond to active breaches.

- Examples:
- Insufficient logging
- Missing alerts for suspicious activities
- Inadequate monitoring
- Local-only logging
- Unclear log messages
- Tamperable logs
- No audit trail for admin actions
- Lack of centralized error reporting


10. A10:2021-Server-Side Request Forgery (SSRF)

Description: SSRF flaws occur when a web application fetches a remote resource without validating the user-supplied URL.

- Examples:
- Unvalidated URL processing
- Missing URL filtering
- Unchecked remote resource access
- Internal services exposure
- Cloud service metadata access
- SSRF via HTTP proxies
- Dynamic link injection