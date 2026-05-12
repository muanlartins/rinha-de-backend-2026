// Package fdpass receives client TCP file descriptors over a Unix
// SEQPACKET socket using the SCM_RIGHTS ancillary data mechanism.
//
// The fd-passing load balancer (e.g. jrblatt/so-no-forevis) accepts client
// connections on :9999, then sendmsg-passes the accepted fd to one of its
// upstream control sockets. The upstream consumes the fd via Receive's
// returned channel and treats it as if it had accept()ed it locally.
//
// See docs/lectures/12-scm-rights.md.
package fdpass
